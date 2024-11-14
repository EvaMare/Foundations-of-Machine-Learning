"""Test the python function from src."""

import sys

import numpy as np

sys.path.insert(0, "./src/")
from src.my_code import Complex, my_function


def test_function() -> None:
    """See it the function really returns true."""
    assert my_function() is True


def test_complex() -> None:
    """Test if the object for a complex number instantiates."""
    complex = Complex(1.0, 2.0)
    assert np.allclose(complex.realpart, 1.0) and np.allclose(complex.imagpart, 2.0)


def test_add() -> None:
    """Test if adding works properly."""
    complex1 = Complex(0.0, 1.0)
    complex2 = Complex(1.0, 0.0)
    complex3 = complex1.add(complex2)
    complex4 = complex3.add(complex2)
    assert np.allclose(complex3.realpart, 1.0) and np.allclose(complex3.imagpart, 1.0)
    assert np.allclose(complex4.realpart, 2.0) and np.allclose(complex4.imagpart, 1.0)


def test_mul() -> None:
    """Test if multiplication works properly."""
    complex1 = Complex(0.0, 1.0)
    complex2 = Complex(1.0, 1.0)
    complex3 = Complex(42.0, 1.0)
    mul = complex1.multiply(complex2)
    mul2 = complex3.multiply(complex2)
    test = (1.0j) * (1 + 1.0j)
    test2 = (42 + 1.0j) * (1 + 1.0j)
    assert np.allclose(mul.realpart, test.real) and np.allclose(mul.imagpart, test.imag)
    assert np.allclose(mul2.realpart, test2.real) and np.allclose(
        mul2.imagpart, test2.imag
    )
