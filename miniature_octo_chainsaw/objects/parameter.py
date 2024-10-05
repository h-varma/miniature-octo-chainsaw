import numpy as np


class Parameter:
    def __init__(self):
        self._name = None
        self._value = None
        self._vary = None
        self._min = None
        self._max = None

        self._true_value = None
        self._is_true_value_set = False

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
    def vary(self):
        return self._vary

    @vary.setter
    def vary(self, vary: bool = False):
        if not isinstance(vary, bool):
            raise TypeError("Vary must be a boolean.")
        self._vary = vary

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

    @property
    def true_value(self):
        return self._true_value

    @true_value.setter
    def true_value(self, true_value: float = None):
        if self._is_true_value_set:
            raise AttributeError("True value is already set.")
        if not true_value:
            raise ValueError("True value cannot be empty.")
        if not isinstance(true_value, (int, float)):
            raise TypeError("True value must be a number.")
        self._true_value = float(true_value)
        self._is_true_value_set = True
