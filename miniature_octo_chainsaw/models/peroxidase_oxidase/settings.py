import numpy as np
from dataclasses import dataclass
from miniature_octo_chainsaw.models.utils import default


@dataclass
class ModelSettings:
    name: str = "peroxidase_oxidase"
    compartments: list = default(["x1", "x2"])
    plot_compartment: str = "x1"

    initial_state: np.ndarray = default(np.array([0, 0]))
    integration_interval: list = default([0, 100])
    non_negative: bool = False

    true_parameters: dict = default({
        "k1": 0.2,
        "k2": 1.75,
        "k3": 2.75,
        "k4": 0.3,
        "k5": 5,
        "k6": 0.14,
    }
    )

    controls: dict = default({"homotopy": "k4", "free": "k5"})
    global_parameters: list = default(["k1", "k2"])

    continuation_settings: dict = default({"h_min": 0, "h_max": 10, "h_step": 0.05})

    bifurcation_type: str = "saddle-node"
    measurement_error: str = "absolute_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
