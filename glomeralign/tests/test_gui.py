import pytest
from glomeralign.gui import load_config, load_brain_and_slices

# Test loading config
def test_load_config():
    config = load_config("glomeralign/config/config.yaml")
    assert "brain_volume_path" in config
    assert "slices_path" in config

# Test loading brain and slices
def test_load_brain_and_slices():
    config = load_config("glomeralign/config/config.yaml")
    brain_volume, slices = load_brain_and_slices(config)
    assert brain_volume is None  # Update once actual loading logic is implemented
    assert isinstance(slices, list)