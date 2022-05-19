import numpy as np

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def arcradians(cos, sin):
    if cos > 0 and sin > 0:
        return np.arccos(cos)
    elif cos > 0 and sin < 0:
        return np.arcsin(sin)
    elif cos < 0 and sin > 0:
        return np.arccos(cos)
    elif cos < 0 and sin < 0:
        return -1 * np.arccos(cos)
