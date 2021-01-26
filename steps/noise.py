from zquantum.core.utils import get_func_from_specs, save_noise_model
from typing import Dict
import yaml
import cirq

def get_cirq_noise_model(noise_specs: Dict):
    """ Creates a simple (pre-baked) Cirq noise model and saves it in an Orquestra compatible json

        Args:
            specs: dictonary containing the following keys:  "module_name", "function_name", and
                noise parameters such as T1, T2 (just T in some cases) and t_gate; all the times
                should be specified in s.
    """
    noise_specs_dict = yaml.load(noise_specs, Loader=yaml.SafeLoader)
    noise_model_func = get_func_from_specs(noise_specs_dict)
    noise_model = noise_model_func(**noise_specs_dict)
    serialized_noise_model = cirq.to_json(noise_model)
    save_noise_model(serialized_noise_model, 'qecirq.noise', 'load_noise_model_from_json', 'noise-model.json')
