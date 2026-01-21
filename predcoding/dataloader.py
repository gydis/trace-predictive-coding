from torch.utils.data import Dataset
import torch

from settings import *

class TraceDataset(Dataset):
    def __init__(self, num_words):
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

