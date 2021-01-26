from cirq import (depolarize, asymmetric_depolarize, 
                  amplitude_damp, phase_damp)
from cirq import to_json, read_json


def get_depolarizing_channel(T, t_gate=10e-9)):
    """Get the depolarizing channel

    Args:
        T (float): Decoherence parameter (seconds)

    """
    gamma = (1 - pow(np.e, - 1/T*t_gate))
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

    px = 0.25*(1- pow(np.e, - t_gate/T_1))
    py = 0.25*(1- pow(np.e, - t_gate/T_1))
    
    exp_1 = pow(np.e, -t_gate/(2*T_1))
    exp_2 = pow(np.e, -t_gate/T_2)
    pz = (0.5 - p_x - 0.5*exp_1*exp_2)

    noise_model = asymmetric_depolarize(p_x = px, p_y=py, p_z = pz)
    return noise_model

def get_amplitude_damping(T_1, t_gate=10e-9):
    """ Creates an amplitude damping noise model
    
    Args:
        T_1 (float) : Relaxation time (seconds)
        t_gate (float) : Discretized time step over which the relaxation occurs over (seconds)
    
    """
    gamma = (1 - pow(np.e, - 1/T_1*t_gate))
    noise_model = amplitude_damp(gamma)
    return noise_model

def get_phase_damping(T_2, t_gate=10e-9):
    """ Creates a dephasing noise model
    
    Args:
        T_2 (float) : dephasing time (seconds)
        t_gate (float) : Discretized time step over which the relaxation occurs over (seconds)

    """

    gamma = (1 - pow(np.e, - 1/T_2*t_gate))
    noise_model = phase_damp(gamma)
    return noise_model

def load_noise_model(filename):
    """Loads a cirq noise model

    Args:
        filename (string): Name of json file that contains the cirq noise model

    Return
        noise_model (cirq.NoiseModel)

    """

    noise_model = read_json(filename)
    return noise_model

def load_noise_model_from_json(serialized_model):
    """Loads a cirq noise model (version 2)

    Args:
        serialized_model (string): json str representation of a cirq noise model

    Return
        noise_model (cirq.NoiseModel)

    """

    noise_model = read_json(json_text = serialized_model)
    return noise_model
