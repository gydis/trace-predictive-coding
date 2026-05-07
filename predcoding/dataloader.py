from typing import Dict, List

from torch.utils.data import Dataset
import torch
import torchaudio
from datasets import load_dataset

from settings import *

class TraceDataset(Dataset):
    def __init__(self, num_words=215, mean_approx=False):
        """
        Dataset for TRACE-like model.

        Parameters
        ----------
        num_words : int
            The number of words in the dataset.
        """
        self.num_words = num_words
        self.mean_approx = mean_approx
        self.words = KNOWN_WORDS[:num_words]
        self.max_word_length = max(len(word) for word in self.words)
        self.words_padded = [word + '-' * (self.max_word_length - len(word)) for word in self.words]
        self.word_features = self._compute_word_features()
        self.word_indices = torch.tensor(
            [WORD_TO_IND[word] for word in self.words],
            dtype=torch.long
        )

    def _compute_word_features(self):
        features = []
        for word in self.words_padded:#if not self.mean_approx else self.words:
            word_feat = []
            for phoneme in word:
                word_feat.append(PHONEMIC_FEATURES[phoneme])
            if self.mean_approx:
                # word_feat = torch.tensor(word_feat, dtype=torch.float32).mean(dim=0, keepdim=True)
                word_feat = [torch.tensor(i, dtype=torch.float32) for i in word_feat]
                word_feat = torch.cat(word_feat, dim=0).unsqueeze(0)
            features.append(torch.tensor(word_feat, dtype=torch.float32))
        return features
    
    def __len__(self):
        return self.num_words

    def __getitem__(self, idx):
        return {
            'word': self.words[idx],
            'features': self.word_features[idx] / 8.0,  # Normalize features to [0, 1]
            'index': self.word_indices[idx],
            'word_padded': self.words_padded[idx]
        }


def trace_collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """Collate TRACE samples by padding to the max word length in the current batch."""
    if len(batch) == 0:
        raise ValueError("trace_collate_fn received an empty batch.")

    words = [sample['word'] for sample in batch]
    lengths = torch.tensor([len(word) for word in words], dtype=torch.long)
    max_batch_word_len = int(lengths.max().item())

    padded_features = []
    padded_words = []
    indices = []
    for sample, word_len in zip(batch, lengths.tolist()):
        features = sample['features']
        if features.ndim != 2:
            raise ValueError(
                f"Expected TRACE features with shape (T, F), got {tuple(features.shape)}."
            )
        if features.shape[0] < word_len:
            raise ValueError(
                "Sample features are shorter than the corresponding word length. "
                "This collate function expects per-phoneme feature rows."
            )

        word_features = features[:word_len]
        if word_len < max_batch_word_len:
            pad = word_features.new_zeros((max_batch_word_len - word_len, word_features.shape[1]))
            word_features = torch.cat([word_features, pad], dim=0)

        padded_features.append(word_features)
        padded_words.append(sample['word'] + '-' * (max_batch_word_len - word_len))
        indices.append(sample['index'])

    return {
        'word': words,
        'features': torch.stack(padded_features, dim=0),
        'index': torch.stack(indices, dim=0).long(),
        'word_padded': padded_words,
        'lengths': lengths,
    }

class PhonemeDataset(Dataset):
    def __init__(self):
        """
        Dataset for phonemes.
        """
        self.phonemes = PHONEMES
        self.phoneme_features = self._compute_phoneme_features()
        self.phoneme_indices = torch.tensor(
            [PHONEME_TO_INDEX[phoneme] for phoneme in self.phonemes],
            dtype=torch.long
        )

    def _compute_phoneme_features(self):
        features = []
        for phoneme in self.phonemes:
            features.append(torch.tensor(PHONEMIC_FEATURES[phoneme], dtype=torch.float32))
        return features
    
    def __len__(self):
        return len(self.phonemes)

    def __getitem__(self, idx):
        return {
            'phoneme': self.phonemes[idx],
            'features': self.phoneme_features[idx],
            'index': self.phoneme_indices[idx]
        }

class SpectrogramMNIST(Dataset):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        time_dim=64,
        return_lengths=False,
        test=False
    ):
        self.ds = load_dataset("gilkeyio/AudioMNIST")["train"] if not test else load_dataset("gilkeyio/AudioMNIST")["test"]
        self.sample_rate = sample_rate
        self.time_dim = time_dim
        self.return_lengths = return_lengths
        self.freq_dim = n_fft // 2 + 1
        
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0  # power spectrogram
        )
        self.cache = {}

        def collate_fn(batch):
            specs, labels = zip(*batch)
            lengths = torch.tensor([spec.shape[0] for spec in specs], dtype=torch.long)

            # Spectrogram transform should keep frequency bins fixed across samples.
            freq_sizes = {spec.shape[1] for spec in specs}
            if len(freq_sizes) != 1:
                raise ValueError(
                    f"Inconsistent spectrogram frequency bins in batch: {sorted(freq_sizes)}"
                )

            target_t = int(lengths.max().item()) if self.time_dim is None else int(self.time_dim)
            batch_specs = []
            for spec in specs:
                t, f = spec.shape
                if t < target_t:
                    pad = spec.new_zeros((target_t - t, f))
                    spec = torch.cat([spec, pad], dim=0)
                elif t > target_t:
                    spec = spec[:target_t]
                batch_specs.append(spec)

            # Final shape: (B, 1, T, F), matching models.spectral_mnist InputLayer.
            specs = torch.stack(batch_specs, dim=0).unsqueeze(1).contiguous()
            labels = torch.tensor(labels, dtype=torch.long)
            lengths = lengths.clamp(max=target_t)
            if self.return_lengths:
                return specs, labels, lengths
            return specs, labels
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        sample = self.ds[idx]

        # 1. Decode audio
        audio_decoder = sample["audio"]
        waveform = audio_decoder.get_all_samples().data  # torch tensor
        sr = audio_decoder.metadata.sample_rate

        # 2. Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # 3. Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 4. Compute spectrogram
        spec = self.spec_transform(waveform)
        spec = torch.log1p(spec).squeeze(0).T  # Remove channel dimension and transpose to (time, freq)
        label = sample["digit"]  # Assuming 'digit' is the label column



        self.cache[idx] = (spec, label)
        return spec, label
