from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelSettings:
    name: str = "minimal_saddle_node"
    data: list = field(default_factory=lambda: [])
    parameters: dict = field(default_factory=lambda: dict())
    compartments: list = field(default_factory=lambda: ["x", "y"])
    plot_compartment: str = "x"

    initial_state: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    integration_interval: list = field(default_factory=lambda: [0, 50])
    non_negative: bool = True

    true_parameters: dict = field(
        default_factory=lambda: {"k1": 3.5, "k2": 1, "k3": 1, "k4": 1}
    )

    controls: dict = field(default_factory=lambda: {"homotopy": "k1", "free": "k3"})
    global_parameters: list = field(default_factory=lambda: ["k4"])

    continuation_settings: dict = field(
        default_factory=lambda: {"h_min": 0.1, "h_max": 10, "h_step": 0.1}
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
    measurement_error: str = "relative_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
