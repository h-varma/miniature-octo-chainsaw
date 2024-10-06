import numpy as np
from dataclasses import dataclass


@dataclass
class ProblemSpecifications:
    name: str

    compartments: list[str]
    to_plot: str

    true_parameters: dict[str, float]

    parameters: dict[str, dict[str, float]]
    parameter_noise: float

    controls: dict[str, str]
    global_parameters: list[str]

    initial_state: np.ndarray
    integration_interval: list[float]    

    bifurcation_type: str
    continuation_settings: dict[str, float]

    measurement_error: str
    data_noise: float
