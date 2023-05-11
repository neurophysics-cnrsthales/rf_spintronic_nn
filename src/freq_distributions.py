import torch
from torch import Tensor
import matplotlib.pyplot as plt


def freq_distribution(distrib_type: str, length: int, minimum: float = 0., maximum: float = 1.,
                      scaling_factor: float = 1.) -> Tensor:
    """Return a tensor corresponding to one out of two frequency distributions: linear or non-linear.

    Args:
        distrib_type (str): String containing the name (type) of frequency distribution (linear or non_linear).
        length (int): Number of samples of the frequency distribution.
        minimum (float): Lower bound of the frequency distribution domain.
        maximum (float): Upper bound of the frequency distribution domain.
        scaling_factor (float): Coefficient of the geometric sequence

    Returns:
        A tensor corresponding to the desired frequency distribution.
    """
    if distrib_type == "linear":
        return torch.linspace(minimum, maximum, length)
    elif distrib_type == "non_linear":
        return non_linear_distrib(minimum, maximum, length, scaling_factor)
    else:
        raise ValueError(f"The {distrib_type} frequency distribution does not exist.")


def non_linear_distrib(minimum: float, maximum: float, length: int, scaling_factor: float) -> Tensor:
    r""" Return a non-linear function based on a geometric sequence, derived by Nathan Leroux [1]:

    .. math::
        f_i = f_0 (\frac{1+\mu}{1-\mu})^i,

    where

    .. math::
        \mu = \frac{\frac{f_{max}}{f_{min}}-1}{\frac{f_{max}}{f_{min}}+1}.

    [1] N.Leroux et al, Phys. Rev. Applied 15, 034067 (2021) https://doi.org/10.1103/PhysRevApplied.15.034067

    Args:
        minimum (float): Lower bound of the frequency distribution domain.
        maximum (float): Upper bound of the frequency distribution domain.
        length (int): Number of samples of the frequency distribution.
        scaling_factor (float): Coefficient of the geometric sequence

    Returns:
        A tensor containing the non-linear function based on a geometric sequence.
    """
    if length > 1:
        coef = (maximum / minimum) ** (1 / (length - 1))
        mu = (coef - 1) / (coef + 1)
        r = (1 + mu) / (1 - mu)
    else:
        raise ValueError(f"The length of frequency distribution must be superior to 1. Got {length} instead.")
    return geometric_sequence(scaling_factor, r, length)


def geometric_sequence(a: float, r: float, n: int) -> Tensor:
    r"""Return a tensor containing a geometric sequence:

    .. math::
        a_n = a_1r^{n-1}.

    Args:
        a (float): Scale factor.
        r (float): Common ratio, a non-zero number.
        n (int): Number of terms of the sequence.

    Returns:
        A tensor containing a geometric sequence.
    """
    return torch.tensor([a * r ** k for k in range(n)])


def visualize_freq_distributions(length, minimum=0.001, maximum=1.):
    fig, ax = plt.subplots()
    distrib_names = ["linear", "non_linear"]
    for distrib in distrib_names:
        ax.plot(freq_distribution(distrib, length, minimum=minimum, scaling_factor=minimum), label=distrib)
    ax.set_xlim(0., length - 1)
    ax.set_ylim(0., maximum + 0.1)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()
