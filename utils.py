import numpy as np
from settings import U, L, R

# Input types numpy arrays of floats
def f(a: np.ndarray, net: np.ndarray, D: float) -> np.ndarray:
    """
    Funcation calculting a change in activation.
    :param a: Current activation.
    :param net: Net input to the unit.
    :param D: Decay rate.
    :return: Change in activation.
    """
    assert all([i == j for i, j in zip(a.shape, net.shape)])
    a_f = a.flatten()
    net_f = net.flatten()
    res = np.zeros_like(a_f)
    mask_less = net_f < 0
    ls = (a_f - L) * net_f - D * (a_f - R)
    gt = (U - a_f) * net_f - D * (a_f - R)
    res[mask_less] = ls[mask_less]
    res[~mask_less] = gt[~mask_less]
    return res.reshape(a.shape)

def get_input_indices(slice_num: int, input_string: str) -> list:
    """
    Produce a 2D list (time slices x 1-2) where for every time slice
    it indicates the number of the input phoneme (by index in input string)
    that should be active at that time slice.
    1 or 2 phonemes can be active at a time slice due to overlap.
    :param slice_num: Number of time slices
    :param input_string: Input string
    :return: 2D list of input phoneme indices
    """
    input_indices = []
    i1 = -1
    i2 = -1
    for i in range(slice_num):
        input_indices.append([])
        if i % 14 == 0: i1 += 1
        if (i-7) % 14 == 0: i2 += 1
        if 14 - (i % 14) > 3 and i1*2 < len(input_string):
            input_indices[-1].append(i1 * 2)
        if i2 >= 0 and 14 - ((i-7) % 14) > 3 and i2*2 + 1 < len(input_string):
            input_indices[-1].append(i2 * 2 + 1)
    return input_indices


slice_to_phoneme_unit_cache = {}
def slice_to_phoneme_unit(slice_idx: int, phoneme_num: int) -> list[int]:
    """
    Function mapping slice index to phoneme unit indices.
    :param slice_idx: Index of the slice.
    :param phoneme_num: Number of phoneme units.
    :return: List of phoneme unit indices active at this slice.
    """
    if (slice_idx, phoneme_num) in slice_to_phoneme_unit_cache:
        return slice_to_phoneme_unit_cache[(slice_idx, phoneme_num)]

    inds = []
    if slice_idx // 6 *2 < phoneme_num:
        inds.append(slice_idx // 6 * 2)
    if slice_idx >= 3 and (slice_idx-3) // 6 * 2 + 1 < phoneme_num:
        inds.append((slice_idx-3) // 6 * 2 + 1)
    slice_to_phoneme_unit_cache[(slice_idx, phoneme_num)] = inds
    return inds

def phoneme_unit_to_slice(unit_idx: int) -> int:
    """
    Function mapping phoneme unit index to slice index.
    :param phoneme_idx: Index of the phoneme.
    :param unit_idx: Index of the unit for this phoneme.
    :return: Slice index corresponding to this phoneme unit's centre.
    """
    if unit_idx % 2 == 0:
        return unit_idx // 2 * 6 + 2
    else:
        return (unit_idx // 2) * 6 + 5