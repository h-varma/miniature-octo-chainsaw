from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelSettings:
    name: str = "peroxidase_oxidase"
    data: list = field(default_factory=lambda: [])
    parameters: dict = field(default_factory=lambda: dict())
    compartments: list = field(default_factory=lambda: ["x1", "x2"])
    plot_compartment: str = "x1"

    initial_state: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    integration_interval: list = field(default_factory=lambda: [0, 100])
    non_negative: bool = False

    true_parameters: dict = field(
        default_factory=lambda: {
            "k1": 0.2,
            "k2": 1.75,
            "k3": 2.75,
            "k4": 0.3,
            "k5": 5,
            "k6": 0.14,
        }
    )

    controls: dict = field(default_factory=lambda: {"homotopy": "k4", "free": "k5"})
    global_parameters: list = field(default_factory=lambda: ["k1", "k2"])

    continuation_settings: dict = field(
        default_factory=lambda: {"h_min": 0, "h_max": 10, "h_step": 0.05}
    )

    mask: dict = field(
        default_factory=lambda: {
            "compartments": False,
            "controls": False,
            "auxiliary_variables": False,
            "global_parameters": False,
        }
    )

    bifurcation_type: str = "saddle-node"
    measurement_error: str = "absolute_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
