from zquantum.core.utils import create_object, save_noise_model
from zquantum.core.typing import Specs
import yaml
import cirq


def get_cirq_noise_model(noise_model_func_specs: Specs, **kwargs):
    """Creates a simple (pre-baked) Cirq noise model and saves it in an Orquestra
        compatible json

    Args:
        specs: dictonary containing the following keys:  "module_name",
            "function_name", and noise parameters such as T1, T2 (just T in some cases)
            and t_gate; all the times should be specified in s.
    """
    if isinstance(noise_model_func_specs, str):
        noise_model_func_specs = yaml.load(
            noise_model_func_specs, Loader=yaml.SafeLoader
        )
    noise_model_func = create_object(noise_model_func_specs)

    noise_model = noise_model_func(**kwargs)
    serialized_noise_model = cirq.to_json(noise_model)
    save_noise_model(
        serialized_noise_model,
        "qecirq.noise",
        "load_noise_model_from_json",
        "noise-model.json",
    )
