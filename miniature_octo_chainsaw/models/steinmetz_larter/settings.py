import numpy as np
from dataclasses import dataclass
from miniature_octo_chainsaw.models.utils import default


@dataclass
class ModelSettings:
    name: str = "steinmetz_larter"
    compartments: list = default(["A", "B", "X", "Y"])
    plot_compartment: str = "A"

    initial_state: np.ndarray = default(np.array([0, 0, 0, 0]))
    integration_interval: list = default([0, 500])
    non_negative: bool = False

    true_parameters: dict = default({
        "k1": 0.1631021,
        "k2": 1250,
        "k3": 0.046875,
        "k4": 20,
        "k5": 1.104,
        "k6": 0.001,
        "k7": 4.24,
        "km7": 0.1175,
        "k8": 0.53,
    }
    )

    controls: dict = default({"homotopy": "k7", "free": "k8"})
    global_parameters: list = default(["k1", "k5"])

    continuation_settings: dict = default({"h_min": -50, "h_max": 50, "h_step": 1}
                                          )

    bifurcation_type: str = "hopf"
    measurement_error: str = "absolute_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
