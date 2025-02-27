# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Test environment utils of MeshPy."""

import os
from unittest.mock import patch

import pytest

from meshpy.utils.environment import (
    cubitpy_is_available,
    get_env_variable,
    is_mybinder,
    is_testing,
)


def test_is_cubitpy_available() -> None:
    """Test is_cubitpy_available function."""

    with patch("importlib.util.find_spec", return_value=True):
        assert cubitpy_is_available() is True

    with patch("importlib.util.find_spec", return_value=None):
        assert cubitpy_is_available() is False


def test_is_mybinder() -> None:
    """Test is_mybinder function."""

    with patch.dict(os.environ, {"BINDER_LAUNCH_HOST": "some_value"}):
        assert is_mybinder() is True

    with patch.dict(os.environ, {}, clear=True):
        assert is_mybinder() is False


def test_is_testing() -> None:
    """Test is_testing function."""

    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "some_value"}):
        assert is_testing() is True

    with patch.dict(os.environ, {}, clear=True):
        assert is_testing() is False


def test_get_env_variable() -> None:
    """Test get_env_variable function."""

    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert get_env_variable("TEST_VAR") == "test_value"

    with patch.dict(os.environ, {}, clear=True):
        assert get_env_variable("TEST_VAR", default="default_value") == "default_value"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError, match="Environment variable TEST_VAR is not set"
        ):
            get_env_variable("TEST_VAR")
