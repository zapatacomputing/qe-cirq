from zquantum.core.utils import get_func_from_specs, save_noise_model
from typing import Dict

def get_cirq_noise_model(specs: Dict):
    """ Creates a simple (pre-baked) Cirq noise model and saves it in an Orquestra compatible json

        Args:
            specs: dictonary containing the following keys:  "module_name", "function_name", and
                noise parameters such as T1, T2 (just T for some cases) and t_gate
    """
    noise_model_func = get_func_from_specs(specs)
    noise_model = noise_model_func(**specs)
    serialized_noise_model = cirq.to_json(noise_model)
    save_noise_model(serialized_noise_model, 'qecirq.noise', 'load_noise_model_from_json')
