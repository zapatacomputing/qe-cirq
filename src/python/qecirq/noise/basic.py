from cirq import depolarize, asymmetric_depolarize, amplitude_damp, phase_damp
from cirq import to_json, read_json
import numpy as np
from typing import Dict, Union
from zquantum.core.utils import save_noise_model
import json


def get_depolarizing_channel(T, t_gate=10e-9):
    """Get the depolarizing channel

    Args:
        T (float): Decoherence parameter (seconds)

    """
    assert T > 0
    assert t_gate > 0

    gamma = 1 - pow(np.e, -1 / T * t_gate)
    noise_model = depolarize(gamma)
    return noise_model


def get_asymmetric_depolarize(T_1, T_2, t_gate=10e-9):
    """Creates a noise model that does both phase and amplitude damping but in the
        Pauli Twirling Approximation discussed the following reference
        https://arxiv.org/pdf/1305.2021.pdf


    Args:
        T_1 (float) : Relaxation time (seconds)
        T_2 (float) : dephasing time (seconds)
        t_gate (float) : Discretized time step over which the relaxation occurs over (seconds)

    """
    assert T_1 > 0
    assert T_2 > 0
    assert t_gate > 0

    px = 0.25 * (1 - pow(np.e, -t_gate / T_1))
    py = 0.25 * (1 - pow(np.e, -t_gate / T_1))

    exp_1 = pow(np.e, -t_gate / (2 * T_1))
    exp_2 = pow(np.e, -t_gate / T_2)
    pz = 0.5 - px - 0.5 * exp_1 * exp_2

    pi = 1 - px - py - pz
    pauli_dict = {"I": pi, "X": px, "Y": py, "Z": pz}

    def make_error_dict_for_circuit(dict1, dict2):
        new_dict = {}
        for key in dict1.keys():
            for key2 in dict2.keys():
                if isinstance(key, str) and isinstance(key2, str):
                    new_dict[key + key2] = dict1[key] * dict2[key2]
        return new_dict

    pta_probabilities = make_error_dict_for_circuit(pauli_dict, pauli_dict)

    # noise_model = asymmetric_depolarize(p_x = px, p_y=py, p_z = pz)
    noise_model = asymmetric_depolarize(error_probabilities=pta_probabilities)
    # to_json(noise_model, 'asymmetric_noise_model.json')
    return noise_model


def get_amplitude_damping(T_1, t_gate=10e-9):
    """Creates an amplitude damping noise model

    Args:
        T_1 (float) : Relaxation time (seconds)
        t_gate (float) : Discretized time step over which the relaxation occurs over (seconds)

    """
    assert T_1 > 0
    assert t_gate > 0

    gamma = 1 - pow(np.e, -1 / T_1 * t_gate)
    noise_model = amplitude_damp(gamma)
    return noise_model


def get_phase_damping(T_2, t_gate=10e-9):
    """Creates a dephasing noise model

    Args:
        T_2 (float) : dephasing time (seconds)
        t_gate (float) : Discretized time step over which the relaxation occurs over (seconds)

    """
    assert T_2 > 0
    assert t_gate > 0
    gamma = 1 - pow(np.e, -1 / T_2 * t_gate)
    noise_model = phase_damp(gamma)
    return noise_model


{
    # noise model info
}


def load_noise_model_from_json(serialized_model: Union[Dict, str]):
    """Loads a cirq noise model (version 2)

    Args:
        serialized_model: json str representation of a cirq noise model

    Return
        noise_model (cirq.NoiseModel)

    """
    if isinstance(serialized_model, dict):
        serialized_model = json.dumps(serialized_model)

    noise_model = read_json(json_text=serialized_model)
    return noise_model


def save_cirq_noise_model(noise_model, filename):
    noise_model_data = to_json(noise_model)
    save_noise_model(
        noise_model_data, "qecirq.noise", "load_noise_model_from_json", filename
    )
