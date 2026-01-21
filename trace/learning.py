import numpy as np
from joblib import Parallel, delayed
from settings import *
from utils import get_input_indices, f, slice_to_phoneme_unit, phoneme_unit_to_slice

def trace(phoneme_string: str, probes: list[tuple[str, int]] = []):

    probe_inds = [(idx, WORD_TO_IND[w]) for w, idx in probes if w in WORD_TO_IND] if probes else [] # (time index, word_index)

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
        feature_values_clipped = np.clip(feature_values, 0, U)
        phoneme_values_clipped = np.clip(phoneme_values, 0, U)

        # Update feature values
        features_net = np.zeros_like(feature_values_clipped)
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
                        features_net[n, dim1, :] -= FEATURE_INHIBITION * feature_values_clipped[n, dim2, :]
        
        # Top-down excitation from phoneme layer
        for ph_idx in range(len(PHONEMES)):
            for unit_idx in range(phoneme_values_clipped.shape[1]):
                unit_centre_slice = phoneme_unit_to_slice(unit_idx)
                window = range(max(unit_centre_slice - 2, 0), min(unit_centre_slice + 4, slice_num))
                for f_idx, feature in enumerate(PHONEMIC_FEATURES[PHONEMES[ph_idx]]):
                    if feature == -1:
                        continue
                    for i,w in enumerate(window):
                        window_weight = i if w < unit_centre_slice else (len(window) - 1 - i)
                        window_weight /= len(window) // 2
                        features_net[w, f_idx, feature] += PHONEME_FEATURE_EXCITATION * window_weight * phoneme_values_clipped[ph_idx, unit_idx]

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

        feature_values_clipped = np.clip(feature_values, 0, U)
        phoneme_values_clipped = np.clip(phoneme_values, 0, U)
        word_values_clipped = np.clip(word_values, 0, U)

        # Phoneme layer delta calculations
        phoneme_net = np.zeros_like(phoneme_values_clipped)
        
        # Bottom-up excitation from feature layer
        for ph_idx in range(len(PHONEMES)):
            for unit_idx in range(phoneme_values_clipped.shape[1]):
                unit_centre_slice = phoneme_unit_to_slice(unit_idx)
                window = range(max(unit_centre_slice - 2, 0), min(unit_centre_slice + 4, slice_num))
                for f_idx, feature in enumerate(PHONEMIC_FEATURES[PHONEMES[ph_idx]]):
                    if feature == -1:
                        continue
                    for i,w in enumerate(window):
                        window_weight = i if w < unit_centre_slice else (len(window) - 1 - i)
                        window_weight /= len(window) // 2
                        phoneme_net[ph_idx, unit_idx] += FEATURE_PHONEME_EXCITATION * window_weight * feature_values_clipped[w, f_idx, feature]
            
        # Lateral inhibition between phoneme units
        for unit_idx1 in range(phoneme_values_clipped.shape[1]):
            match unit_idx1:
                case 0:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1])
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1 + 1])

                case _ if unit_idx1 == phoneme_values_clipped.shape[1] - 1:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1])
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1 - 1])

                case _:
                    for phoneme_idx in range(len(PHONEMES)):
                        phonemes_mask = np.ones(len(PHONEMES), dtype=bool)
                        phonemes_mask[phoneme_idx] = False
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1])
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1 - 1])
                        phoneme_net[phoneme_idx, unit_idx1] -= PHONEME_INHIBITION * 0.5 * np.sum(phoneme_values_clipped[phonemes_mask, unit_idx1 + 1])

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
                        phoneme_net[phoneme_idx, unit] += WORD_PHONEME_EXCITATION * weight * word_values_clipped[n, w]

        phoneme_delta = f(phoneme_values, phoneme_net, PHONEME_DECAY)

        return phoneme_delta

    def word_layer_update_delta(word_values: np.ndarray,
                                phoneme_values: np.ndarray,
                                slice_num: int,
                                curr_slice: int = -1) -> np.ndarray:
        """
        Function updating word layer activations for step i.
        :param word_values: Current word layer activations.
        :param phoneme_values: Current phoneme layer activations.
        :param slice_num_i: Current slice number.
        :param slice_num: Total number of slices.
        :return: Word layer activations delta for the time slice.
        """
        phoneme_values_clipped = np.clip(phoneme_values, 0, U)
        word_values_clipped = np.clip(word_values, 0, U)

        word_net = np.zeros_like(word_values)
        # Word layer delta calculations

        # Lateral inhibition between words (vectorized)
        # For every (start_slice, word) pair we have an interval [start, end).
        # Compute pairwise overlaps and apply inhibition in a single matrix multiply.
        # Note: this builds a (P x P) overlap matrix where P = slice_num * WORDS_NUM.
        # durations = np.array([len(w) * 6 for w in KNOWN_WORDS], dtype=int)  # (WORDS_NUM,)
        # # Flattened indices for all (n, w) combinations
        # starts = np.repeat(np.arange(slice_num, dtype=int), WORDS_NUM)  # (P,)
        # word_idx = np.tile(np.arange(WORDS_NUM, dtype=int), slice_num)   # (P,)
        # ends = np.minimum(starts + durations[word_idx], slice_num)      # (P,)
        # min_durs = durations[word_idx]                                   # (P,)

        # P = starts.size
        # if P > 0:
        #     # Pairwise overlap: overlap[i, j] = max(0, min(end_i, end_j) - max(start_i, start_j))
        #     starts_col = starts[:, None]
        #     ends_col = ends[:, None]
        #     starts_row = starts[None, :]
        #     ends_row = ends[None, :]

        #     overlap = np.maximum(0, np.minimum(ends_col, ends_row) - np.maximum(starts_col, starts_row))

        #     # Normalize by the minimum duration of the two words involved
        #     min_len = np.minimum(min_durs[:, None], min_durs[None, :])
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         overlap_coef = np.zeros_like(overlap, dtype=float)
        #         valid = min_len > 0
        #         overlap_coef[valid] = overlap[valid] / min_len[valid]

        #     # Exclude self-contribution (i == j)
        #     np.fill_diagonal(overlap_coef, -WORD_SELF_EXCITATION)

        #     # Flatten clipped word values and multiply
        #     word_clip = np.maximum(0, word_values_clipped.ravel())  # (P,)
        #     # For each target i, inhibition is WORD_INHIBITION * sum_j overlap_coef[i,j] * word_clip[j]
        #     inhibition_vec = WORD_INHIBITION * (overlap_coef @ word_clip**2)
        #     inhibition_matrix = inhibition_vec.reshape(slice_num, WORDS_NUM)
        #     word_net -= inhibition_matrix
        word_pool = np.zeros_like(word_values_clipped)
        for n in range(slice_num):
            for w in range(WORDS_NUM):
                word = KNOWN_WORDS[w]
                duration = len(word) * 6
                start = n
                end = min(start + duration, slice_num)
                word_pool[n, w] += (word_values_clipped[start:end, :]**2).sum() - WORD_SELF_EXCITATION * (word_values_clipped[start:end, w]**2).sum()
        word_net -= WORD_INHIBITION * word_pool
        
        # Phoneme to word excitation
        weights = [0, 1/3, 2/3, 1, 2/3, 1/3, 0]
        for n in range(slice_num):
            for w in range(WORDS_NUM):
                word = KNOWN_WORDS[w]
                center_slices = [n + 6*i for i in range(len(word))]
                slices = [(slice-i, weights[i]) for i in range(-3, 4) for slice in center_slices]
                slices = list(filter(lambda x: 0 <= x[0] < slice_num, slices))
                # if (n, w) in probe_inds:
                #     print(f"Time: {curr_slice}, updated_slice: {n}")
                #     print(f"\tCenter slices: {slices}")
                phoneme_units = [slice_to_phoneme_unit(slice[0], PHONEME_NUM) for slice in slices]
                for weight, units, phoneme in zip([slice[1] for slice in slices], phoneme_units, word):
                    for unit in units:
                        phoneme_idx = PHONEME_TO_INDEX[phoneme]
                        word_net[n, w] += PHONEME_WORD_EXCITATION * weight * phoneme_values_clipped[phoneme_idx, unit]

        # Update word activations
        word_delta = f(word_values, word_net, WORD_DECAY)
        return word_delta

    feature_values_agg = []
    phoneme_values_agg = []
    word_values_agg = []
    probe_values = np.zeros((len(probe_inds), slice_num))
    for i in range(slice_num):

        # Compute phoneme and word deltas in parallel
        feature_values_delta, phoneme_values_delta, word_values_delta = Parallel(n_jobs=-1)(
            [
                delayed(feature_layer_update_delta)(feature_values, phoneme_values, input_string, i, slice_num),
                delayed(phoneme_layer_update_delta)(feature_values, phoneme_values, word_values, slice_num),
                delayed(word_layer_update_delta)(word_values, phoneme_values, slice_num, i)
            ]
        )

        feature_values += feature_values_delta
        feature_values = np.clip(feature_values, L, U)
        feature_values_agg.append(feature_values.copy())

        phoneme_values += phoneme_values_delta
        phoneme_values = np.clip(phoneme_values, L, U)
        phoneme_values_agg.append(phoneme_values.copy())

        word_values += word_values_delta
        word_values = np.clip(word_values, L, U)
        word_values_agg.append(word_values.copy())

        if probe_inds:
            probe_values[:, i] = word_values.flat[np.ravel_multi_index(np.array(probe_inds).T, word_values.shape)]
                
    return feature_values_agg, phoneme_values_agg, word_values_agg, probe_values
