import math
from typing import Union

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from freq_distributions import freq_distribution


def instanciate_model(model_name, model_params, device=torch.device("cpu")):
    if model_name == "MLP":
        model = MLP(**model_params).to(device)

    elif model_name == "spinMLP":
        model = spinMLP(**model_params).to(device)

    else:
        raise Exception("No model found, please use either MLP or spinMLP.")
    return model


class MLP(nn.Module):
    r"""Multilayered perceptron

    Args:
        network_size (tuple):
            Tuple containing the number of input features, the size of each hidden layers and the
            number of output features.
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        nb_layers (int): Number of layers.
        nb_hidden_layers (int): Number of hidden layers.
        layers (ModuleList): List of nn.Linear modules.
        activations (ModuleList): List of nn.ReLU modules.

    """

    def __init__(self, network_size, bias: bool = True) -> None:
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.network_size = network_size
        self.in_features = network_size[0]
        self.out_features = network_size[-1]
        self.nb_layers = len(network_size) - 1
        self.nb_hidden_layers = self.nb_layers - 1
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f, bias=bias) for in_f, out_f in zip(network_size, network_size[1:])])
        self.activations = nn.ModuleList([nn.ReLU() for i in range(self.nb_hidden_layers)] + [nn.Identity()])

        self.outputs = [0 for i in range(self.nb_layers)]
        self.outputs_act = [0 for i in range(self.nb_layers)]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, input_powers: Tensor):
        x = input_powers.reshape(-1, self.in_features)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.outputs[i] = x
            x = self.activations[i](x)
            self.outputs_act[i] = x
        return x


class spinMLP(nn.Module):
    r""" The physical multi-layer perceptron. We assume that the number of resonators of the hidden layers is the
    same as the number of input hidden nodes (aka oscillators) for each layer. We also assume that each hidden layers
    have an activation function corresponding to the formula for the output power of the spin-torque nano-oscillators
    (:func:`~activation_functions.stno`). This formula requires a current as an input, meaning that we need to convert
    the output voltages from the chains of resonators (with an additional voltage bias) to currents by using a
    voltage-to-current ratio :math:`g_\text{m}`:

    .. math::
        I = (V^{\text{chains}} + V^{\text{layer}}) g_\text{m}

    In our case, we consider that the voltages unit is :math:`mV` and the voltage-to-current ratio is in :math:`\mu
    A/mV`. Each activation function converts the chains output voltages (previous layer) into powers. For each layers
    (except the last one), the power is amplified to reach a maximum of one. In fact, we consider that the input
    power (for each frequency) for each layer is preserved along each resonator chain. In reality, there is a power
    loss, but we consider implicitly that this loss is compensated by some artificial amplification.

    Args:
        network_size (tuple): Tuple containing the number of input frequencies for the first layer, the size of each
         hidden layers and the number of output features
        input_freq (Tensor): Input frequencies
        resonators_params (dict): Dictionary containing all the parameters of the resonators for the class
         :func:`~LayerChainsResonators`
        oscillators_params (dict): Dictionary containing all the parameters of the oscillators for the class
         :func:`~LayerOscillators`
        freq_res_bounds ([list[list[float]]]): A list containing the minimum and maximum frequencies for each layer.
         Example: [[0.020,0.120], [0.020,0.120]]
        freq_res_distrib (list[str]): List of string containing the type of resonance frequency distributions we use for
         each layer
        add_voltage_bias (list): Additional voltage bias (used for all chains).
        voltage_to_current_factors (list): Conversion factor (voltage_to_current_factors gm) (converting an input
         voltage into a current)
        is_with_nonidealities (bool): If True, modify the output power with :func:`~add_nonidealities`.

    Attributes:
        out_features (int): Number of output features
        nb_layers (int): Number of layers of chains of resonators
        nb_hidden_layers (int): Number of hidden layers of chains of resonators.
        layers: List of LayerChainsResonators instantiations
        nb_input_freq (int): Number of input frequencies
        input_freq (Tensor): Input frequencies
        layers_voltage_biases (Tensor): Additional voltage bias (used for all chains).
        voltage_to_current_factors (Tensor): Conversion factor (voltage_to_current_factors gm) (converting an input voltage into
         a current)
        activations (): Activation functions associated to each layer. The last activation function (output layer)
        correspond to the identity.
        V_to_mV: scaling voltage from V to mV

    """

    def __init__(self, network_size: tuple, input_freq: Tensor, resonators_params,
                 oscillators_params, nb_input_resonators, freq_res_bounds: list[list[float]],
                 freq_res_distrib: list[str], add_voltage_bias: list, voltage_to_current_factors: list,
                 is_with_nonidealities: bool = False):
        super(spinMLP, self).__init__()
        self.name = 'spinMLP'
        self.network_size = network_size

        self.in_features = len(input_freq)
        self.nb_layers = len(network_size) - 1
        self.nb_hidden_layers = self.nb_layers - 1
        self.out_features = network_size[-1]

        self.nb_input_freq = len(input_freq)
        self.input_freq = Parameter(input_freq, requires_grad=False)

        self.freq_res_bounds = freq_res_bounds
        self.V_to_mV = 1e3  # scaling voltage from V to mV

        self.layers_voltage_biases = nn.Parameter(torch.tensor(add_voltage_bias), requires_grad=False)
        self.voltage_to_current_factors = nn.Parameter(torch.tensor(voltage_to_current_factors), requires_grad=False)
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        res_chains_size = tuple([nb_input_resonators] + list(network_size[1:]))
        for i, (nb_resonators_per_chain, nb_chains) in enumerate(zip(res_chains_size, res_chains_size[1:])):
            self.layers.append(LayerChainsResonators(nb_resonators_per_chain, nb_chains,
                                                     **resonators_params,
                                                     freq_res_bounds=freq_res_bounds[i],
                                                     freq_res_distrib=freq_res_distrib[i],
                                                     is_with_nonidealities=is_with_nonidealities))
            self.activations.append(LayerOscillators(nb_chains, **oscillators_params,
                                                     is_with_nonidealities=is_with_nonidealities))

        # workaround modification frequencies of the first layer
        self.layers[0].set_input_freq(self.input_freq)
        # Workaround override the last layer with Identity (No oscillators for the last layer)
        self.activations[-1] = nn.Identity()

        self.voltages = [0. for i in range(self.nb_layers)]
        self.currents = [0. for i in range(self.nb_layers)]
        self.outputs = [0. for i in range(self.nb_layers)]

    def reset_parameters(self, add_voltage_bias, voltage_to_current_factors):
        """Reset the learning parameters of each layer and set the physical hyper-parameters"""
        self.layers_voltage_biases.data = torch.tensor(add_voltage_bias + [0.])
        self.voltage_to_current_factors.data = torch.tensor(voltage_to_current_factors + [1.])
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, input_powers: Tensor):
        """

        Args:
            input_powers (Tensor): The powers corresponds to the input dataset values

        Returns:
            Scores or logits of the spintronic network corresponding to voltages.
        """
        x = input_powers.reshape(-1, len(self.input_freq))  # * 1e-6
        for i, layer in enumerate(self.layers):
            x = (layer(x) + self.layers_voltage_biases[i]) * self.V_to_mV
            x = x * self.voltage_to_current_factors[i]  # output currents
            x = self.activations[i](x)
        return x


class LayerChainsResonators(nn.Module):
    r"""Create a set of chains of resonators acting as a neural network layer.

    Given an input RF current as an input of a resonator, the resonator will induce a voltage rectification given by
    an expression obtained that corresponds to the analytic model of the paper. The complete expression of the
    voltage rectification of the resonator :math:`k` of the chain :math:`j` submitted to the input signal :math:`i` is

    .. math::
        v_{ijk} = (-1)^k P_i G_{ijk}

    where the factor :math:`(-1)^k` take into account the fact that the resonators inside a chain are connected
    alternatively in series, :math:`P_i` is the power of the input signal :math:`i`, :math:`G` is the rectification
    tensor:

    .. math::
        G_{ijk}= \frac{2\alpha f_{jk}^{\text{res}} (f_i^{\text{in}}-f_{jk}^{\text{res}})}{(\alpha  \
        f_{jk}^{\text{res}})^2+(f_i^{\text{in}}-f_{jk}^{\text{res}})^2} K_{SD}

    where :math:`f_i^{\text{in}}` is the frequency of the input signal :math:`i` and :math:`f_{jk}^{\text{res}}` is the
    frequency of the resonator :math:`k` of the chain :math:`j` at resonance and :math:`\alpha` is the Gilbert damping.

    Finally,

    .. math::
        K_{SD} = \frac{\Delta R}{R}\frac{1}{I_{\text{th}}}\frac{\tan \gamma_p}{8\sqrt{2}}\beta_s

    where it is assumed that the TMR :math:`\frac{\Delta R}{R}\approx 1`, the shape factor :math:`\beta_s\approx 1` and
    :math:`\frac{\tan \gamma_p}{8\sqrt{2}}\approx 0.088` (:math:`\gamma_p=\pi/4`).


    Note:
        - For convenience, we can change the connection between resonators by choosing either :math:`(-1)^{k}` or
          :math:`(-1)^{k+1}`.
        - For the input powers of physical MLP, it is expected that :math:`P_i\in [0,1]` where 1 corresponds to
          :math:`1\, \mu W`.
        - For the frequencies :math:`f_i^{\text{in}}` and :math:`f_{jk}^{\text{res}}` the order of magnitude is the
          **GHz**.
        - It is expected that :math:`I_{th}\approx 10\mu A`. Since the order of magnitude of both power and
          current is the same (:math:`\mu W` and :math:`\mu A`) they are compensated, meaning we don't need to write the
          factors of the order of magnitude.

    .. _rectification:

    In this model, it is normally assumed that the signal fed in the resonator has only one frequency component (need
    check). But it seems (need ref) that the total voltage rectification of one resonator of an analog input signal
    corresponds to the integral over the "frequency domain of rectification" of the analytic model [1]. While in
    practice this domain have a finite length, in the model it has an infinite length. We need to keep that in mind.
    Therefore, given an RF current as an input of a chain of resonators, each resonator will rectify the input
    signal. In our case, the input signal is either a real analog signal, e.g. an RF signal, or an artificial signal
    made of an arbitrary set of frequencies, e.g. the pixels of an image encoded into one RF current (MLP case) or
    into different field lines interacting independently with each resonators of a chain (CNN case).


    Since we assumed that a resonator can rectify a signal composed of several frequency components, which is the
    case for an analog signal, the weight matrix is computed using

    .. math::
        W_{ji} = \sum_k(-1)^k G_{jik}

    .. _output_voltages:

    Finally, the output voltages of all chains are obtained through the linear transformation of the incoming signals:

    .. math::
        V = PW^T + b

    where :math:`P` corresponds to the input powers vector, :math:`W^T` is the transpose of weight matrix and
    :math:`b` is the vector containing the bias voltages vector. This transpose operation is necessary because we are
    using the Pytorch class torch.nn.functional.linear.

    Note:
        More explanation on the way to write the rectification function and the weight matrix can be found in [1].


    Args:
        nb_resonators_per_chain (int): Number of resonator inside one chain (chains supposed same length). It needs to
         be the same as the number of input frequencies if we do not consider overlap.
        nb_chains (int): Number of chains of resonators. It also corresponds to the number of outputs.
        weight_scaling (float): Scaling factor of the **weight** attribute.
        bias_scaling (float): Scaling factor for the **bias** attribute.


    Attributes:
        weight (Tensor):
            The learnable weights of the module with a shape :math:`(\text{nb_chains},
            \text{nb_resonators_per_chain})` that correspond to the shift with respect to the initial resonance
            frequencies of all resonators. Alternatively, it can correspond directly to the resonance frequencies of all
            resonators (for that need to uncomment a line in init_initial_res_freq).
        bias (Tensor):
            The learnable biases of the module of shape :math:`(\text{nb_chains})`. Physically, it corresponds to
            voltage biases (one per chain).
            If :attr:`bias` is ``True``, the values are initialized from
            :math:`\mathcal{N}(0, \sigma)` where :math:`\sigma = \frac{0.01}{\sqrt{\text{nb_chains}}}`.
        initial_res_freq (Tensor): The initial resonance frequencies of a chain (same for all chains).
        input_freq (Tensor): The input frequencies of the layer. By default, it corresponds to the initial
         freq_distribution of resonance frequencies. It can be modified by using the set_input_freq method.
        resonators_connection (Tensor):The factor that characterises the connection between the resonators of a chain. For a
         head-to-tail arrangement, it corresponds to the :math:`(-1)^k` factor (see N.Leroux et al, Phys. Rev.Applied 15
         ,034067 (2021)).
         freq_res_min (float): The initial minimum resonance frequency of all chains (associated to the first resonator
          of each chain).
         freq_res_max (float): The initial maximum resonance frequency of all chains (associated to the last resonator of
          each chain).
         damping (float): The Gilbert damping parameter.
         Ith (float): The threshold current of the MTJ. While it exists in the case of excitation of a STNO, such
          threshold effect do not exist for the case of a resonator. However, it appears in the calculation of
          rectification.
         KSD (float): The spin torque sensitivity.
    """

    def __init__(self, nb_resonators_per_chain: int, nb_chains: int, bias: bool = True,
                 freq_res_bounds: list[float] = None,
                 signed_connection: str = "k+1", Ith_res: float = 10., freq_res_distrib: str = "non_linear",
                 weight_scaling: float = 1e-3, bias_scaling: float = 1e-2, damping: float = 0.01,
                 is_with_nonidealities: bool = False,
                 freq_var_percentage: float = 0.01) -> None:

        super(LayerChainsResonators, self).__init__()
        self.nb_resonators_per_chain = nb_resonators_per_chain
        self.nb_chains = nb_chains
        self.out_features = nb_chains

        self.freq_res_min = freq_res_bounds[0]
        self.freq_res_max = freq_res_bounds[1]
        self.freq_res_distrib = freq_res_distrib
        self.initial_res_freq = Parameter(freq_distribution(self.freq_res_distrib, self.nb_resonators_per_chain,
                                                            minimum=self.freq_res_min, maximum=self.freq_res_max,
                                                            scaling_factor=self.freq_res_min), requires_grad=False)
        self.input_freq = Parameter(freq_distribution(self.freq_res_distrib, self.nb_resonators_per_chain,
                                                      minimum=self.freq_res_min, maximum=self.freq_res_max,
                                                      scaling_factor=self.freq_res_min), requires_grad=False)
        self.connection = signed_connection
        self.resonators_connection = Parameter(torch.empty(nb_resonators_per_chain), requires_grad=False)
        self.init_resonators_connection()

        # Frequency Deviations from the initial resonance frequencies (learning parameters)
        self.weight = Parameter(torch.empty(nb_chains, nb_resonators_per_chain), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(nb_chains), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.weight_scaling = 1 / np.sqrt(nb_resonators_per_chain) * weight_scaling
        self.bias_scaling = 1 / np.sqrt(nb_chains) * bias_scaling
        self.damping = damping
        self.Ith = Ith_res  # MicroAmp  #e-6 #Amp
        self.KSD = math.tan(math.pi / 4) / (8 * math.sqrt(2)) / self.Ith  # Spin-diode sensitivity (not in SI units)
        self.reset_parameters()

        if is_with_nonidealities:
            self.add_nonidealities(self.input_freq, freq_var_percentage)
            self.add_nonidealities(self.initial_res_freq, freq_var_percentage)

    def set_input_freq(self, input_freq) -> None:
        self.input_freq = Parameter(input_freq, requires_grad=False)

    def add_nonidealities(self, parameter, var_percentage) -> None:
        r"""Modify the input tensor t using a sample :math:`X` from a normal distribution :math:`\mathcal{N}(0,1)` scaled
        by the variability percentage :math:`\varepsilon` using the formula:

        .. math::
            t^{'} = t(1+X\varepsilon).
        """
        rand_values = torch.randn_like(parameter.data) * var_percentage
        parameter.data = parameter.data * (1 + rand_values)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.weight_scaling)
        if self.bias is not None:
            nn.init.normal_(self.bias, std=self.bias_scaling)

    def init_resonators_connection(self) -> None:
        if self.connection == "k":
            self.resonators_connection.data = torch.Tensor(
                [(-1) ** k for k in range(self.nb_resonators_per_chain)])
        elif self.connection == "k+1":
            self.resonators_connection.data = torch.Tensor(
                [(-1) ** (k + 1) for k in range(self.nb_resonators_per_chain)])

    def rectification(self):
        input_freq = self.input_freq.data.reshape(1, len(self.input_freq), 1)
        res_freq = self.initial_res_freq + self.weight
        res_freq = res_freq.reshape(self.nb_chains, 1, self.nb_resonators_per_chain)
        W = 2 * self.damping * res_freq * (input_freq - res_freq) / (
                (self.damping * res_freq) ** 2 + (input_freq - res_freq) ** 2) * self.KSD * self.resonators_connection
        return torch.sum(W, dim=2)

    def forward(self, input_powers: Tensor) -> Tensor:
        r"""Performs a "linear" transformation on the input powers.

        Args:
            input_powers:
                The input powers associated to the inputs signals. 1 unit power = :math:`1\, \mu W`.

        Returns:
            The product between the input powers and the rectification matrix whose elements {Wij} corresponds to the
            rectification function of the chain j submitted to an input signal i. Hence, it corresponds to a
            "vector" containing the :ref:`rectification voltages <output_voltages>` of all chains.
        """
        return F.linear(input_powers, self.rectification(), self.bias)

    def extra_repr(self) -> str:
        return 'nb_resonators_per_chain={}, nb_chains={}, bias={}'.format(
            self.nb_resonators_per_chain, self.nb_chains, self.bias is not None
        )


class LayerOscillators(nn.Module):
    r"""Create a set of spin-transfer nano-oscillators (STNOs) corresponding to activation functions.
    The expression of the normalized output power of one STNO is:

    .. math::
        p = |c|^2 = \frac{\xi - 1}{\xi + Q} \quad (\text{if } \xi > 1, 0 \, \text{otherwise})

    where :math:`c` is the normalized amplitude of the stationary precession, :math:`\xi=I_{
    \text{DC}}/I_{\text{th}}` and :math:`Q` is the non-linear damping coefficient. In order to have a non-null output
    power, the input DC current :math:`I_{\text{DC}}` must be superior to a threshold current :math:`I_{\text{th}}`.
    The scaled output power is given by

    .. math::
        P = A \, p(I_{DC}) \,(\frac{\Delta R}{R}\beta )^2 \, R I_{DC}^2

    where :math:`A` is a scaling factor, :math:`\frac{\Delta R}{R}` is the TMR, :math:`\beta` is the shape
    factor and :math:`R` is the resistance of all the oscillators. We assume that the TMR is 100%
    (:math:`\frac{\Delta R}{R}=1)` and the shape factor equal to 1 (:math:`\beta \approx 1`) which reduces the
    expression to

    .. math::
        P = A \, p(I_{DC}) \, R\, I_{DC}^2

    Since the output powers cannot be negative, we clamped all the values of current inferior to the threshold current
    (including negative values) to :math:`I_{\text{th}}`. This gives :math:`\xi = 1` leading to a normalized power of
    0, thus an output power of 0. In addition to this, the currents are clamped at :math:`4 I_{th}` to mimic the
    experimental prevention of the destruction of the oscillators.

    The variability of the output powers is implemented as a modification of the scaling factor using the method
    :func:`~add_nonidealities`.

    Args:
        nb_chains (int): Number of input chains of resonators
        Q (float): Non-linear damping coefficient. The value has to be positive (default: 2).
        Ith_osc (float): Threshold current (current necessary to get auto-oscillations).
        Iclamp (float): Current value above which the current is clamped.
        R_osc (float): Oscillators resistance.
        scaling (float): Scaling factor used to prevent vanishing gradient. This is due to the fact that the inputs
         values decrease while propagating in the network. It corresponds to a change of units from watt to microwatt
         which is the unit expected for the input of a LayerChainsResonators instance.
        amp_factor (float): It corresponds to the action of a power amplifier that adjust the output power of the
         oscillators.
        is_with_nonidealities (bool): If True, modify the output power with :func:`~add_nonidealities`.
        power_var_percentage (float): Variability amplitude corresponding to the standard deviation of the normal
         distribution from which the samples are extracted (see :func:`~add_nonidealities`)
    Note:
        It is expected that :math:`Q=2`, :math:`I_{th}\, \approx 10\mu A` and :math:`R_{osc}\approx 1k\Omega`.
    """

    def __init__(self, nb_chains, Q=2., Ith_osc=10, Iclamp=40, R_osc=1e3, scaling=1e-6, amp_factor=1.25,
                 is_with_nonidealities=False, power_var_percentage=0.01):
        super(LayerOscillators, self).__init__()
        self.Q = Q
        self.Ith_osc = Ith_osc  # microA
        self.Iclamp = Iclamp  # microA
        self.R_osc = R_osc  # Ohm
        self.is_with_nonidealities = is_with_nonidealities

        if is_with_nonidealities:
            amp = torch.ones(nb_chains)  # This contains percentages of the resulting power found later on.
            self.add_nonidealities(amp, power_var_percentage)
            amp = torch.clamp(amp, min=0.) * amp_factor
        else:
            amp = torch.tensor(amp_factor)
        self.factor = Parameter(scaling * amp, requires_grad=False)
        # self.visualize()

    def add_nonidealities(self, parameter, var_percentage) -> None:
        r"""Modify the input tensor t using a sample :math:`X` from a normal distribution :math:`\mathcal{N}(0,1)`
        scaled by the variability percentage :math:`\varepsilon` using the formula:

        .. math::
            t^{'} = t(1+X\varepsilon).
        """
        rand_values = torch.randn_like(parameter.data) * var_percentage
        parameter.data = parameter.data * (1 + rand_values)

    def normalized_stno_power(self, input_currents):
        r"""Return the normalized output powers of a set of STNOs

        .. math::
            p = \frac{\xi - 1}{\xi + Q} \quad (\text{if } \xi > 1, 0 \, \text{otherwise})

        """
        xi = input_currents / self.Ith_osc
        return (xi - 1) / (xi + self.Q)

    def forward(self, input_currents):
        r"""Return the output powers of a set of STNOs.

        .. math::
            P = A \, p(I_{DC}) \, R\, I_{DC}^2
        """
        input_currents = torch.clamp(input_currents, max=self.Iclamp, min=self.Ith_osc)
        output_power = self.normalized_stno_power(input_currents) * input_currents ** 2 * self.R_osc
        return output_power * self.factor
