import numpy as np
import matplotlib.pyplot as plt
from settings import U, L, PHONEMIC_FEATURES, PHONEMES, WORD_TO_IND, KNOWN_WORDS
import pandas as pd

def plot_feature_responses(phoneme: str, feature_values_agg, slice_num: int):
    avg_b_responses = np.zeros((slice_num, slice_num), dtype=np.float64)
    for t in range(slice_num):
        feature_vals = feature_values_agg[t]
        b_features = PHONEMIC_FEATURES[phoneme]
        response = 0.0
        for f_idx, feature in enumerate(b_features):
            if feature == -1:
                continue
            response += feature_vals[:, f_idx, feature]
        avg_b_responses[t] = response / (7 - np.sum(b_features == -1))
    plt.imshow(avg_b_responses, aspect='auto', cmap='hot')
    plt.colorbar(label=f'Average activation for phoneme "{phoneme}" features')
    plt.clim(L, U)
    plt.xlabel('Time slice for the input showing')
    plt.ylabel('Time slices of feature activations')
    plt.show()

def plot_phoneme_responses(phoneme: str, phoneme_values_agg, slice_num: int):
    phoneme_responses = np.zeros((slice_num, phoneme_values_agg[0].shape[1]), dtype=np.float64)
    for t in range(slice_num):
        phoneme_vals = phoneme_values_agg[t]
        phoneme_idx = PHONEMES.index(phoneme)
        phoneme_responses[t] = phoneme_vals[phoneme_idx]
    plt.figure()
    plt.imshow(phoneme_responses.T, aspect='auto', cmap='hot')
    plt.colorbar(label=f'Activation for phoneme "{phoneme}" units')
    plt.clim(L, U)
    plt.ylabel(f'Phoneme "{phoneme}" unit index')
    plt.xlabel('Time slice for the input showing')
    plt.title(f'Activation of phoneme "{phoneme}" units over time')
    plt.show()

def plot_word_unit_responses(word, word_values_agg, slice_num: int, input_string: str):
    word_responses = np.zeros((slice_num, word_values_agg[0].shape[0]), dtype=np.float64)
    for t in range(slice_num):
        word_vals = word_values_agg[t]
        word_responses[t] = word_vals[:, WORD_TO_IND[word]]
    plt.figure(figsize=(16,16))
    plt.imshow(word_responses.T, aspect='auto', cmap='hot')
    plt.colorbar(label=f'Activation for word "{word}" units')
    plt.clim(L, U)
    # overlay numerical values on the heatmap
    mat = word_responses.T  # shape: (n_units, slice_num) as displayed
    n_rows, n_cols = mat.shape
    # adaptive font size for readability
    fontsize = 6
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            plt.text(j, i, f"{val*100:2.0f}", ha='center', va='center', color="black", fontsize=fontsize)
    plt.ylabel(f'Word "{word}" unit index')
    plt.xlabel('Time slice for the input showing')
    plt.title(f'Activation of word "{word}" units over time with input "{input_string}"')
    plt.show()

    max_activation_word_indices = np.argmax(word_values_agg[-1], axis=1)
    activations = []
    for i in range(slice_num):
        activations.append(word_values_agg[-1][i, max_activation_word_indices[i]])
    activations = np.array(activations)
    probs = np.exp(activations) / np.sum(np.exp(activations))
    print(probs.shape)
    words = [KNOWN_WORDS[idx] for idx in max_activation_word_indices]
    time = list(range(slice_num))
    words_df = pd.DataFrame({'Time': time, 'Max Activation Word': words, 'Probability': probs})
    words_df = words_df.sort_values(by='Probability', ascending=False)
    return words_df

def plot_probe_responses(probe_values, probes):
    plt.figure()
    for i, (word, time_idx) in enumerate(probes):
        plt.plot(probe_values[i], label=f'Word unit "{word}" at slice {time_idx}')
    plt.xlabel('Time tick for the input showing')
    plt.ylabel('Activation value')
    plt.title('Probe word activations over time')
    plt.legend()
    plt.show()