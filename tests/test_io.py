"""
 Created By:   Jason Herning
 File Name:    test_io.py
 Date Created: 01-25-2021
 Description:  Tests for functions that handle file read/write.
"""

from os import getcwd
from pathlib import Path

import numpy as np
import pytest
from insar_eventnet.io import load_dataset, save_dataset


@pytest.fixture(scope="function")
def supply_datadir_cwd(monkeypatch):
    """Fixture to patch current working directory to datadir."""
    monkeypatch.chdir(getcwd())
    test_make_data_path = Path("tests/test_io").resolve()
    print(test_make_data_path)
    monkeypatch.chdir(test_make_data_path)


dataset_1_wrapped = dataset_1_unwrapped = np.eye(25)


@pytest.mark.usefixtures("supply_datadir_cwd")
def test_load_dataset():
    """Test that loaded arrays equal correct value"""
    unwrapped, wrapped = load_dataset(Path("test_dataset_1.npz"))
    assert np.array_equal(unwrapped, dataset_1_unwrapped)
    assert np.array_equal(wrapped, dataset_1_wrapped)


@pytest.mark.usefixtures("supply_datadir_cwd")
def test_load_dataset_fail():
    """Test that loaded arrays don't == wrong value."""
    unwrapped, wrapped = load_dataset(Path("test_dataset_1.npz"))
    assert not np.array_equal(unwrapped, np.full((25, 25), 9))
    assert not np.array_equal(wrapped, np.eye(24))


@pytest.mark.usefixtures("supply_datadir_cwd")
def test_save_dataset():
    """Test saving the dataset to .npz file."""
    test_wrap = np.full((40, 40), 9)
    test_unwrap = np.eye(40)

    save_dataset(Path("new_dataset"), unwrapped=test_unwrap, wrapped=test_wrap)
    assert len(list(Path(".").rglob("new_dataset.npz"))) == 1


@pytest.mark.usefixtures("supply_datadir_cwd")
def test_save_load_dataset():
    """Test both saving then loading the dataset"""
    test_wrap = np.full((17, 17), 1)
    test_unwrap = np.eye(17)

    save_dataset(Path("new_test_dataset"), unwrapped=test_unwrap, wrapped=test_wrap)
    unwrapped, wrapped = load_dataset(Path("new_test_dataset.npz"))

    assert np.array_equal(unwrapped, test_unwrap)
    assert np.array_equal(wrapped, test_wrap)
