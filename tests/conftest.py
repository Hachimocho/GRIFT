import os
import random
import numpy as np
import pytest

# Ensure deterministic random for tests
random.seed(42)
np.random.seed(42)

@pytest.fixture(autouse=True)
def _chdir_tmpdir(tmp_path, monkeypatch):
    # Run tests from a temp dir to avoid polluting repo with artifacts
    monkeypatch.chdir(tmp_path)

