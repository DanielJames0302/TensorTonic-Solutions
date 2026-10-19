import numpy as np



def sigmoid(x):
    """Computes the element-wise sigmoid of an input array or scalar."""
    return 1 / (1 + np.exp(-x))

def discriminator(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Returns discriminator probabilities as a float64 array with shape (B, 1).
    """
    return sigmoid(x @ W)