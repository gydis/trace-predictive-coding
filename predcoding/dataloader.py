from torch.utils.data import Dataset
import torch

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