from torch.utils.data import Dataset
import torch

from settings import *

class TraceDataset(Dataset):
    def __init__(self, num_words=215):
        """
        Dataset for TRACE-like model.

        Parameters
        ----------
        num_words : int
            The number of words in the dataset.
        """
        self.num_words = num_words
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
        for word in self.words_padded:
            word_feat = []
            for phoneme in word:
                word_feat.append(PHONEMIC_FEATURES[phoneme])
            features.append(torch.tensor(word_feat, dtype=torch.float32))
        return features
    
    def __len__(self):
        return self.num_words

    def __getitem__(self, idx):
        return {
            'word': self.words[idx],
            'features': self.word_features[idx],
            'index': self.word_indices[idx]
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