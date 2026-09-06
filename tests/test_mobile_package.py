"""Installed cross-repository dependency proof; no Android commands execute."""
import json
import subprocess
import sys

def test_installed_sdk_boundary_from_real_vigil_cli():
    from beast_studio_client import mobile
    assert mobile.SCHEMA == "beast.android-observation/v1"
    result = subprocess.run([sys.executable, "-m", "perception.mobile_watch",
        "--serial", "test-device", "--package", "com.example.test",
        "--adb", "missing-adb-proof"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "ADB unavailable" in json.loads(result.stdout)["error"]
