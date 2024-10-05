import numpy as np
from dataclasses import dataclass
from miniature_octo_chainsaw.models.utils import default


@dataclass
class ProblemSpecs:
    name: str = "predator_prey"
    compartments: list = default(["N", "C", "R", "B"])
    plot_compartment: str = "N"

    initial_state: np.ndarray = default(np.array([6, 5, 1, 4]))
    integration_interval: list = default([0, 50])
    non_negative: bool = False

    true_parameters: dict = default({
        "delta": 0.05,
        "Ni": 160,
        "Bc": 3.3,
        "Kc": 4.3,
        "Bb": 2.25,
        "Kb": 15,
        "epsilon": 0.25,
        "m": 0.055,
        "lambda": 0.4,
    }
    )

    controls: dict = default({"homotopy": "Ni", "free": "delta"})

    global_parameters: list = default(["Bc", "Bb"])

    continuation_settings: dict = default({"h_min": 10, "h_max": 1000, "h_step": 20}
                                          )

    bifurcation_type: str = "hopf"
    measurement_error: str = "absolute_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
