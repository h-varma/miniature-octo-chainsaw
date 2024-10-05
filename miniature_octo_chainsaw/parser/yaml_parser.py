import yaml
from miniature_octo_chainsaw.logging_ import logger


class YamlParser:
    def __init__(self, file_path):
        self.model_spec_path = file_path + "/model_specifications.yaml"
        self.problem_spec_path = file_path + "/problem_specifications.yaml"
        self.model_specifications = None
        self.problem_specifications = None

    def load_yaml(self, file_path):
        try:
            logger.info(f"Loading YAML file: {file_path}")
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                logger.info(f"Successfully loaded YAML file.")
                return data
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

    def parse_model_specifications(self):
        logger.info("Parsing model specifications.")
        self.model_specifications = self.load_yaml(self.model_spec_path)

    def parse_problem_specifications(self):
        logger.info("Parsing problem specifications.")
        self.problem_specifications = self.load_yaml(self.problem_spec_path)

    def get_model_specifications(self):
        if self.model_specifications is None:
            logger.info("Model specifications not loaded yet. Parsing now.")
            self.parse_model_specifications()
        return self.model_specifications

    def get_problem_specifications(self):
        if self.problem_specifications is None:
            logger.info("Problem specifications not loaded yet. Parsing now.")
            self.parse_problem_specifications()
        return self.problem_specifications
