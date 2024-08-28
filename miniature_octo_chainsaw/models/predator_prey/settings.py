from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelSettings:
    name: str = "predator_prey"
    data: list = field(default_factory=lambda: [])
    parameters: dict = field(default_factory=lambda: dict())
    compartments: list = field(default_factory=lambda: ["N", "C", "R", "B"])
    plot_compartment: str = "N"

    initial_state: np.ndarray = field(default_factory=lambda: np.array([6, 5, 1, 4]))
    integration_interval: list = field(default_factory=lambda: [0, 50])
    non_negative: bool = False

    true_parameters: dict = field(
        default_factory=lambda: {
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

    controls: dict = field(default_factory=lambda: {"homotopy": "Ni", "free": "delta"})

    global_parameters: list = field(default_factory=lambda: ["Bc", "Bb"])

    continuation_settings: dict = field(
        default_factory=lambda: {"h_min": 10, "h_max": 1000, "h_step": 20}
    )

    mask: dict = field(
        default_factory=lambda: {
            "compartments": False,
            "controls": False,
            "auxiliary_variables": False,
            "global_parameters": False,
        }
    )

    bifurcation_type: str = "hopf"
    measurement_error: str = "absolute_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
