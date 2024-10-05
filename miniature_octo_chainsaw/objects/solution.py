import numpy as np

class Solution:
    def __init__(self, states=None, controls=None, parameters=None, auxiliary=None):
        self._model_states = states if states is not None else []
        self._controls = controls if controls is not None else []
        self._parameters = parameters if parameters is not None else []
        self._auxiliary = auxiliary if auxiliary is not None else []
        
        self._toggles = {
            'states': True,
            'controls': True,
            'parameters': True,
            'auxiliary': True
        }
    
    @property
    def states(self):
        return self._model_states
    
    @states.setter
    def states(self, value):
        self._model_states = value
    
    @property
    def controls(self):
        return self._controls
    
    @controls.setter
    def controls(self, value):
        self._controls = value
    
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
    
    @property
    def auxiliary(self):
        return self._auxiliary
    
    @auxiliary.setter
    def auxiliary(self, value):
        self._auxiliary = value
    
    @property
    def toggles(self):
        return self._toggles
    
    def toggle_component(self, component, state):
        if component in self._toggles:
            self._toggles[component] = state
        else:
            raise ValueError(f"Component {component} does not exist.")
    
    def to_numpy(self):
        arrays = []
        for key in self._toggles:
            if self._toggles[key]:
                arrays.append(np.array(getattr(self, key)))
        return np.hstack(arrays) if arrays else np.array([])
    
    def from_numpy(self, array):
        index = 0
        for key in self._toggles:
            if self._toggles[key]:
                value = getattr(self, key)
                length = len(value)
                setattr(self, key, array[index:index+length].tolist())
                index += length
    
    def to_dict(self):
        return {key: getattr(self, key) for key in self._toggles if self._toggles[key]}
    
    def from_dict(self, dictionary):
        for key, value in dictionary.items():
            if key in self._toggles:
                setattr(self, key, value)
            else:
                raise ValueError(f"Component {key} does not exist.")

# Example usage:
# solution = Solution(states=[1, 2, 3], controls=[4, 5], parameters=[6], auxiliary=[7, 8])
# solution.toggle_component('controls', False)
# np_array = solution.to_numpy()
# solution.from_numpy(np_array)
# dict_representation = solution.to_dict()
# solution.from_dict(dict_representation)