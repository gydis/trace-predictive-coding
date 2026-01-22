import numpy as np

# Parameters
U = 1.0                           # Upper bound on activations
L = -0.3                          # Lower bound on activations
R = 0                             # Resting value of activations
FEATURE_DECAY = 0.01              # Decay rate of activations
PHONEME_DECAY = 0.03              # Decay rate of activations
WORD_DECAY = 0.05                 # Decay rate of activations
FEATURE_INHIBITION = 0.04         # Inhibition between features
PHONEME_FEATURE_EXCITATION = 0.00 # Top-down excitation of feature level from the phoneme
FEATURE_PHONEME_EXCITATION = 0.02 # Bottom-up excitation of phoneme level from the features
PHONEME_INHIBITION = 0.08         # Inhibition between phonemes
WORD_INHIBITION = 0.03            # Inhibition between words
PHONEME_WORD_EXCITATION = 0.08    # Bottom-up excitation of word level from the phonemes
WORD_PHONEME_EXCITATION = 0    # Top-down excitation of phoneme level from the words
WORD_SELF_EXCITATION = 0.1       # Self-excitation of words

with open('wordlist', 'r') as f:
    KNOWN_WORDS = [line.strip() for line in f.readlines()]
# KNOWN_WORDS = KNOWN_WORDS[:100]
# KNOWN_WORDS = ["bar", "aba", "ark", "bark"]
WORD_TO_IND = {word: i for i, word in enumerate(KNOWN_WORDS)} # Mapping from words to indices
WORDS_NUM = len(KNOWN_WORDS) # Number of words in the lexicon

# Phonemes and their features
PHONEMES = ['p','b','t','d','k','g','s','S','r','l','a','i','u','^','-']
PHONEMIC_FEATURES = {
    'p' : [4, 1, 7, 2, 8, 1, 8],
    'b' : [4, 1, 7, 2, 8, 7, 7],
    't' : [4, 1, 7, 7, 8, 1, 6],
    'd' : [4, 1, 7, 7, 8, 7, 5],
    'k' : [4, 1, 2, 3, 8, 1, 4],
    'g' : [4, 1, 2, 3, 8, 7, 3],
    's' : [6, 4, 7, 8, 5, 1, 0],
    'S' : [6, 4, 6, 4, 5, 1, 0],
    'r' : [7, 7, 1, 2, 3, 8, 0],
    'l' : [7, 7, 2, 4, 3, 8, 0],
    'a' : [8, 8, 2, 1, 1, 8, 0],
    'i' : [8, 8, 8, 8, 1, 8, 0],
    'u' : [8, 8, 6, 2, 1, 8, 0],
    '^' : [7, 8, 5, 1, 1, 8, 0],
    '-' : [0, 0, 0, 0, 0, 0, 0]
}
# PHONEMIC_FEATURES = {k: np.array(v, dtype=int) - 1 for k, v in PHONEMIC_FEATURES.items()}  # Shift features to start from 0
PHONEME_TO_INDEX = {phoneme: i for i, phoneme in enumerate(PHONEMES)}