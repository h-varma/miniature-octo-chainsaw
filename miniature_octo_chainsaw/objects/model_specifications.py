from dataclasses import dataclass
from miniature_octo_chainsaw.objects.states import States
from miniature_octo_chainsaw.objects.parameter import Parameter


@dataclass
class ModelSpecifications:
    model_name: str
    
    compartments: list[States]
    compartment_to_plot: str

    parameters: list[Parameter]
    


