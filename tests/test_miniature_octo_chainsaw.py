import autograd.numpy as np
from miniature_octo_chainsaw.models.importer import import_model


def test_model_correctness():
    model = import_model(name="predator_prey")
    model.mask["compartments"] = True

    model.parameters = {'delta': {'value': 0.04, 'vary': False},
                        'Ni': {'value': 80, 'vary': False},
                        'Bc': {'value': 3.3, 'vary': False},
                        'Kc': {'value': 4.3, 'vary': False},
                        'Bb': {'value': 2.25, 'vary': False},
                        'Kb': {'value': 15, 'vary': False},
                        'epsilon': {'value': 0.25, 'vary': False},
                        'm': {'value': 0.055, 'vary': False},
                        'lambda': {'value': 0.4, 'vary': False}}

    assert model.mask["compartments"] is True
    assert model.mask["controls"] is False
    assert model.mask["global_parameters"] is False
    assert model.mask["auxiliary_variables"] is False

    x = np.array([0, 0, 0, 0], dtype=float)
    rhs_ = model.rhs_(x)
    jacobian_ = model.jacobian_(x)
    assert np.allclose(rhs_, np.array([3.2, 0, 0, 0]))
    assert np.allclose(jacobian_, np.array([[-0.04, 0, 0, 0],
                                            [0, -0.04, 0, 0],
                                            [0, 0, -0.495, 0],
                                            [0, 0, 0, -0.095]]))

    x = np.array([1, 1, 1, 1], dtype=float)
    rhs_ = model.rhs_(x)
    jacobian_ = model.jacobian_(x)
    assert np.allclose(rhs_, np.array([2.537358490566038,
                                       0.020141509433962236,
                                       -0.354375,
                                       0.045625]))
    assert np.allclose(jacobian_, np.array([[-0.5451619793520827, -0.6226415094339622, 0, 0],
                                            [0.5051619793520826, 0.055297759433962236, 0, -0.5625],
                                            [0, 0.1318359375, -0.354375, 0],
                                            [0, 0.1318359375, 0.140625, -0.095]]))
