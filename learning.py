import numpy as np
from joblib import Parallel, delayed
from settings import *
from utils import get_input_indices, f, slice_to_phoneme_unit, phoneme_unit_to_slice

def trace(phoneme_string: str):

    input_string = phoneme_string  # Input string
    slice_num = 11 + 7 * (len(input_string)-1) # Number of time slices

    input_indices = get_input_indices(slice_num, input_string)

    PHONEME_NUM = slice_num // 3 - 1
    feature_values = np.zeros((slice_num, 7, 8), dtype=np.float64) # (time, dimension, value)
    phoneme_values = np.zeros((len(PHONEMES), PHONEME_NUM), dtype=np.float64) # (phoneme, number of unit)
    word_values = np.zeros((slice_num, WORDS_NUM), dtype=np.float64) # (time, word)

    def feature_layer_update_delta(feature_values: np.ndarray,
                                phoneme_values: np.ndarray,
                                input_string: str,
                                slice_num_i: int,
                                slice_num: int) -> np.ndarray:
        """
        Function updating feature layer activations for step i.
        :param feature_values: Current feature layer activations.
        :param phoneme_values: Current phoneme layer activations.
        :param input_string: Input string of phonemes.
        :param slice_num_i: Current slice number.
        :param slice_num: Total number of slices.
        :return: Feature layer activations delta for the time slice.
        """
        i = slice_num_i

        # Update feature values
        features_net = np.zeros_like(feature_values)
        # Input excitation
        for ind in input_indices[i]:
            phoneme = input_string[ind]
            features = PHONEMIC_FEATURES[phoneme]

            for f_idx, feature in enumerate(features):
                if feature == -1:
                    continue
                features_net[i, f_idx, feature] += 1.0
                
            
        # Lateral inhibition
        for n in range(slice_num):
            for dim1 in range(7):
                for dim2 in range(7):
                    if dim1 != dim2:
                        features_net[n, dim1, :] -= FEATURE_INHIBITION * np.clip(feature_values[n, dim2, :], 0, U)
        
        # Top-down excitation from phoneme layer
        for ph_idx in range(len(PHONEMES)):
            for unit_idx in range(phoneme_values.shape[1]):
                unit_centre_slice = phoneme_unit_to_slice(unit_idx)
                window = range(max(unit_centre_slice - 2, 0), min(unit_centre_slice + 4, slice_num))
                for f_idx, feature in enumerate(PHONEMIC_FEATURES[PHONEMES[ph_idx]]):
                    if feature == -1:
                        continue
                    for i,w in enumerate(window):
                        window_weight = i if w < unit_centre_slice else (len(window) - 1 - i)
                        window_weight /= len(window) // 2
                        features_net[w, f_idx, feature] += PHONEME_FEATURE_EXCITATION * window_weight * np.clip(phoneme_values[ph_idx, unit_idx], 0, U)

        # Update feature activations
        features_delta = f(feature_values, features_net, FEATURE_DECAY)
        return features_delta

    def phoneme_layer_update_delta(feature_values: np.ndarray,
                                phoneme_values: np.ndarray,
                                word_values: np.ndarray,
                                slice_num: int) -> np.ndarray:
        """
        Function updating phoneme layer activations for step i.
        :param phoneme_values: Current phoneme layer activations.
        :param feature_values: Current feature layer activations.
        :param slice_num_i: Current slice number.
        :param slice_num: Total number of slices.
        :return: Phoneme layer activations delta for the time slice.
        """
        # Phoneme layer delta calculations
        phoneme_net = np.zeros_like(phoneme_values)
        
        # Bottom-up excitation from feature layer
        for ph_idx in range(len(PHONEMES)):
            for unit_idx in range(phoneme_values.shape[1]):
                unit_centre_slice = phoneme_unit_to_slice(unit_idx)
                window = range(max(unit_centre_slice - 2, 0), min(unit_centre_slice + 4, slice_num))
                for f_idx, feature in enumerate(PHONEMIC_FEATURES[PHONEMES[ph_idx]]):
                    if feature == -1:
                        continue
                    for i,w in enumerate(window):
                        window_weight = i if w < unit_centre_slice else (len(window) - 1 - i)
                        window_weight /= len(window) // 2
                        phoneme_net[ph_idx, unit_idx] += FEATURE_PHONEME_EXCITATION * window_weight * np.clip(feature_values[w, f_idx, feature], 0, U)
            
        # Lateral inhibition between phoneme units
        for unit_idx1 in range(phoneme_values.shape[1]):
            match unit_idx1:
                case 0:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1], 0, U))
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1 + 1], 0, U))

                case _ if unit_idx1 == phoneme_values.shape[1] - 1:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1], 0, U))
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1 - 1], 0, U))

                case _:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1], 0, U))
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1 - 1], 0, U))
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(np.clip(phoneme_values[phonemes_mask, unit_idx1 + 1], 0, U))

        # Top-down excitation from word layer
        for n in range(slice_num):
            for w in range(WORDS_NUM):
                word = KNOWN_WORDS[w]
                center_slices = [n + 6*i for i in range(len(word))]
                slices = [(i-1, 0.5) for i in center_slices] + [(i, 1) for i in center_slices] + [(i+1, 0.5) for i in center_slices]
                slices = list(filter(lambda x: 0 <= x[0] < slice_num, slices))
                phoneme_units = [slice_to_phoneme_unit(slice[0], PHONEME_NUM) for slice in slices]
                for weight, units, phoneme in zip([slice[1] for slice in slices], phoneme_units, word):
                    for unit in units:
                        phoneme_idx = PHONEME_TO_INDEX[phoneme]
                        phoneme_net[phoneme_idx, unit] += WORD_PHONEME_EXCITATION * weight * np.clip(word_values[n, w], 0, U)

        phoneme_delta = f(phoneme_values, phoneme_net, PHONEME_DECAY)

        return phoneme_delta

    def word_layer_update_delta(word_values: np.ndarray,
                                phoneme_values: np.ndarray,
                                slice_num: int) -> np.ndarray:
        """
        Function updating word layer activations for step i.
        :param word_values: Current word layer activations.
        :param phoneme_values: Current phoneme layer activations.
        :param slice_num_i: Current slice number.
        :param slice_num: Total number of slices.
        :return: Word layer activations delta for the time slice.
        """
        # Word layer delta calculations
        # Lateral inhibition between words
        for n1 in range(slice_num):
            for w1 in range(WORDS_NUM):
                window_end = min(len(KNOWN_WORDS[w1]) * 6 + n1, slice_num)
                window_start = n1
                for n2 in range(slice_num):
                    for w2 in range(WORDS_NUM):
                        if w1 != w2 or n1 != n2:
                            window_end2 = min(len(KNOWN_WORDS[w2]) * 6 + n2, slice_num)
                            window_start2 = n2
                            overlap = max(0, min(window_end, window_end2) - max(window_start, window_start2))
                            overlap_coef = overlap / min(len(KNOWN_WORDS[w1]) * 6, len(KNOWN_WORDS[w2]) * 6)
                            if overlap_coef > 0:
                                word_values[n1, w1] -= WORD_INHIBITION * overlap_coef * np.clip(word_values[n2, w2], 0, U)
        
        # Phoneme to word excitation
        for n in range(slice_num):
            for w in range(WORDS_NUM):
                word = KNOWN_WORDS[w]
                center_slices = [n + 6*i for i in range(len(word))]
                slices = [(i-1, 0.5) for i in center_slices] + [(i, 1) for i in center_slices] + [(i+1, 0.5) for i in center_slices]
                slices = list(filter(lambda x: 0 <= x[0] < slice_num, slices))
                phoneme_units = [slice_to_phoneme_unit(slice[0], PHONEME_NUM) for slice in slices]
                for weight, units, phoneme in zip([slice[1] for slice in slices], phoneme_units, word):
                    for unit in units:
                        phoneme_idx = PHONEME_TO_INDEX[phoneme]
                        word_values[n, w] += PHONEME_WORD_EXCITATION * weight * np.clip(phoneme_values[phoneme_idx, unit], 0, U)

        # Update word activations
        word_delta = f(word_values, word_values, WORD_DECAY)
        return word_delta

    feature_values_agg = []
    phoneme_values_agg = []
    word_values_agg = []
    for i in range(slice_num):

        # Compute phoneme and word deltas in parallel
        feature_values_delta, phoneme_values_delta, word_values_delta = Parallel(n_jobs=-1)(
            [
                delayed(feature_layer_update_delta)(feature_values, phoneme_values, input_string, i, slice_num),
                delayed(phoneme_layer_update_delta)(feature_values, phoneme_values, word_values, slice_num),
                delayed(word_layer_update_delta)(word_values, phoneme_values, slice_num)
            ]
        )

        word_values_delta = word_layer_update_delta(word_values, phoneme_values, slice_num)

        feature_values += feature_values_delta
        feature_values = np.clip(feature_values, L, U)
        feature_values_agg.append(feature_values.copy())

        phoneme_values += phoneme_values_delta
        phoneme_values = np.clip(phoneme_values, L, U)
        phoneme_values_agg.append(phoneme_values.copy())

        word_values += word_values_delta
        word_values = np.clip(word_values, L, U)
        word_values_agg.append(word_values.copy())
                
    return feature_values_agg, phoneme_values_agg, word_values_agg
