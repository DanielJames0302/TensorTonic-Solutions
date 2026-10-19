import numpy as np

def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    N = real_probs.shape[0]
    e = 10 ** -8
    clipped_real_probs = np.clip(real_probs, e, 1 - e)
    clipped_fake_probs = np.clip(fake_probs, e, 1 - e)
    loss_d = float(np.sum(np.log(clipped_real_probs) + np.log(1 - clipped_fake_probs)) * (-1/N))
    loss_g = float(np.sum(np.log(clipped_fake_probs)) * (-1/N))
    result = {
        "discriminator_loss": loss_d,
        "generator_loss": loss_g
    }
    return result