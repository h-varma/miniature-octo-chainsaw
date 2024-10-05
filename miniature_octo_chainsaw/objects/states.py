import numpy as np


class State:
    def __init__(self):
        self._name = None
        self._value = None
        self._min = None
        self._max = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        if not name:
            raise ValueError("Name cannot be empty.")
        if not isinstance(name, str):
            raise TypeError("Name must be a string.")
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        if not value:
            raise ValueError("Value cannot be empty.")
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number.")
        self._value = float(value)

    @property
    def lower_bound(self):
        return self._min

    @lower_bound.setter
    def lower_bound(self, min_: float = -np.inf):
        if not isinstance(min_, (int, float)):
            raise TypeError("Min must be a number.")
        self._min = float(min_)

    @property
    def upper_bound(self):
        return self._max

    @upper_bound.setter
    def upper_bound(self, max_: float = np.inf):
        if not isinstance(max_, (int, float)):
            raise TypeError("Max must be a number.")
        self._max = float(max_)


class States:
    def __init__(self):
        self._states = []

    def add_state(self, state: State):
        self._states.append(state)


