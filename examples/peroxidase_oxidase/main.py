import sys
import os

# Add the root directory of the project to the Python path
folder_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(folder_path, '../..')))

import pandas as pd
# from model_equations import Model
from miniature_octo_chainsaw.parser.yaml_parser import YamlParser
from miniature_octo_chainsaw.logging_ import logger


# model = Model()
data = pd.read_table(folder_path + "/data.dat", sep=" ")

# model.generate_parameter_guesses()
# logger.info(f"True model parameters: {model.true_parameters}")
# logger.info(f"Parameter guess initialization: {model.parameters}")

parser = YamlParser(file_path=folder_path)
problem_specifications = parser.get_problem_specifications()