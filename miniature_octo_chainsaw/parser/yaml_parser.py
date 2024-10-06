import yaml
import numpy as np
from dataclasses import dataclass
from ..models.specifications import ProblemSpecifications
from ..logging_ import logger


class YamlParser:
    def __init__(self, file_path):
        self.model_specs_path = file_path + "/model_specifications.yaml"
        self.meta_params_path = file_path + "/meta_parameters.yaml"
        self.model_specifications = None
        self.meta_parameters = None
        self.specifications = None

    def __load_yaml(self, file_path):
        try:
            logger.info(f"Loading YAML file: {file_path}")
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                logger.info(f"Successfully loaded YAML file.")
                return data
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

    def __load_model_specifications(self):
        """
        Load the model specifications from the YAML file.

        Returns
        -------
        dict: The model specifications
        """
        logger.info("Parsing model specifications.")
        self.model_specifications = self.__load_yaml(self.model_specs_path)

    def __load_meta_parameters(self):
        """
        Load the meta parameters from the YAML file.

        Returns
        -------
        dict: The meta parameters
        """
        logger.info("Parsing meta parameters.")
        self.meta_parameters = self.__load_yaml(self.meta_params_path)

    def __parse_problem_specifications(self) -> ProblemSpecifications:
        """
        Parse the model specifications from the YAML file.

        Parameters
        ----------
        model_specs: dict
            The model specifications.

        Returns
        -------
        ModelSpecifications: The parsed model specifications.
        """
        kwargs = {}
        kwargs["name"] = self.model_specifications["model_name"]

        kwargs["compartments"] = []
        kwargs["initial_state"] = []
        kwargs["true_parameters"] = {}
        kwargs["parameters"] = {}

        for key, values in self.model_specifications.items():
            if key.startswith("compartment"):
                kwargs["compartments"].append(values["name"])
                kwargs["initial_state"].append(values["value"])

            elif key.startswith("parameter"):
                p_name = values["name"]
                true_p = values["true_value"]
                kwargs["true_parameters"][p_name] = true_p

                try:
                    initial_p = values["initial_value"]
                except KeyError:
                    initial_p = true_p
                kwargs["parameters"][p_name] = {"value": initial_p, "vary": False}

        kwargs["to_plot"] = self.model_specifications["to_plot"]

        assert (
            kwargs["name"] == self.meta_parameters["model_name"]
        ), "Model name in model specifications and meta parameters do not match."

        kwargs["data_noise"] = self.meta_parameters["data_noise"]["scale"]
        kwargs["parameter_noise"] = self.meta_parameters["parameter_noise"]["scale"]

        kwargs["integration_interval"] = np.array([0, self.meta_parameters["t_end"]])
        kwargs["bifurcation_type"] = self.meta_parameters["bifurcation"]["type"]

        control_1 = self.meta_parameters["control_1"]
        control_2 = self.meta_parameters["control_2"]

        kwargs["controls"] = {"homotopy": control_1["name"], "free": control_2["name"]}

        kwargs["global_parameters"] = self.meta_parameters["global_parameters"]

        kwargs["continuation_settings"] = {
            "h_min": control_1["min_value"],
            "h_max": control_1["max_value"],
            "h_step": control_1["step_size"],
        }

        residual = self.meta_parameters["residual"]
        kwargs["measurement_error"] = residual["type"] + "_" + residual["scale"]

        return ProblemSpecifications(**kwargs)

    def get_problem_specifications(self):
        """
        Get all the problem specifications from the relevant YAML files.

        Returns
        -------
        dict: The problem specifications.
        """
        if self.model_specifications is None:
            logger.info("Model specifications not loaded yet. Parsing now.")
            self.__load_model_specifications()

        if self.meta_parameters is None:
            logger.info("Meta parameters not loaded yet. Parsing now.")
            self.__load_meta_parameters()

        self.specifications = self.__parse_problem_specifications()
        return self.specifications
