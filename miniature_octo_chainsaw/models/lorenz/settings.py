from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelSettings:
    name: str = "lorenz"
    data: list = field(default_factory=lambda: [])
    parameters: dict = field(default_factory=lambda: dict())
    compartments: list = field(default_factory=lambda: ["x", "y", "z"])
    plot_compartment: str = "x"

    initial_state: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    integration_interval: list = field(default_factory=lambda: [0, 100])
    non_negative: bool = True

    true_parameters: dict = field(
        default_factory=lambda: {"sigma": 10, "r": 12.8563, "b": 0.7590}
    )

    controls: dict = field(default_factory=lambda: {"homotopy": "r", "free": "b"})
    global_parameters: list = field(default_factory=lambda: ["sigma"])

    continuation_settings: dict = field(
        default_factory=lambda: {"h_min": 0, "h_max": 20, "h_step": 0.5}
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
    measurement_error: str = "relative_linear"
    data_noise: float = 0.05
    parameter_noise: float = 1
