"""Installed cross-repository dependency proof; no Android commands execute."""
import json
import subprocess
import sys

import pytest


def test_installed_sdk_boundary_from_real_vigil_cli():
    pytest.importorskip("beast_studio_client.mobile")
    result = subprocess.run([sys.executable, "-m", "perception.mobile_watch",
        "--serial", "test-device", "--package", "com.example.test",
        "--adb", "missing-adb-proof"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "ADB unavailable" in json.loads(result.stdout)["error"]
