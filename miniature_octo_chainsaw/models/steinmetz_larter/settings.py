from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelSettings:
    name: str = "steinmetz_larter"
    data: list = field(default_factory=lambda: [])
    parameters: dict = field(default_factory=lambda: dict())
    compartments: list = field(default_factory=lambda: ["A", "B", "X", "Y"])
    plot_compartment: str = "A"

    initial_state: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0]))
    integration_interval: list = field(default_factory=lambda: [0, 500])
    non_negative: bool = False

    true_parameters: dict = field(
        default_factory=lambda: {
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

    controls: dict = field(default_factory=lambda: {"homotopy": "k7", "free": "k8"})
    global_parameters: list = field(default_factory=lambda: ["k1", "k5"])

    continuation_settings: dict = field(
        default_factory=lambda: {"h_min": -50, "h_max": 50, "h_step": 1}
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
