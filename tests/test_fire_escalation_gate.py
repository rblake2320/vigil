"""Finite real local-record tests. No models, camera, network or GPU."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("vigil_demo_gate", Path(__file__).resolve().parents[1] / "core/vigil_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def reply(**changes):
    value = {"event": "fire_colors", "confirmed": True, "summary": "Orange candidate"}
    value.update(changes)
    return json.dumps(value)


@pytest.mark.parametrize("raw", [
    "Yes fire", "VLM unavailable: timeout", "null", "[]", "{", 
    reply(confirmed="true"), reply(confirmed=1), reply(confirmed=None),
    reply(event="person_in_zone"), reply(extra=True), reply(summary=""),
    '{"event":"fire_colors","confirmed":false,"confirmed":true,"summary":"x"}',
])
def test_invalid_review_cannot_reach_sink(tmp_path, raw):
    sink = tmp_path / "sink.json"
    cascade = demo.ActionCascade(tmp_path / "records", test_sink=sink)
    event = cascade.fire("fire_colors", "HIGH", raw, np.zeros((8, 8, 3), dtype=np.uint8))
    assert event["review"] == "refused"
    assert not sink.exists()
    assert json.loads(cascade.log_path.read_text())["review"] == "refused"


def test_false_refuses(tmp_path):
    sink = tmp_path / "sink"
    event = demo.ActionCascade(tmp_path / "records", sink).fire("fire_colors", "HIGH", reply(confirmed=False), np.zeros((8,8,3), dtype=np.uint8))
    assert event["review"] == "refused" and not sink.exists()


def test_default_ignores_webhook_and_records(tmp_path, monkeypatch):
    monkeypatch.setenv("VIGIL_WEBHOOK", "https://example.invalid/emergency")
    def forbidden(*args, **kwargs):
        raise AssertionError("network not allowed")
    monkeypatch.setattr(demo.urllib.request, "urlopen", forbidden)
    cascade = demo.ActionCascade(tmp_path)
    event = cascade.fire("fire_colors", "HIGH", reply(), np.zeros((8,8,3), dtype=np.uint8))
    assert event["review"] == "candidate_confirmed"
    assert event["mode"] == "record_only" and event["test_sink"] == "not_attempted"
    assert json.loads(cascade.log_path.read_text()) == event
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_optin_sink_one_record_restart_no_overwrite(tmp_path):
    sink = tmp_path / "sink"
    frame = np.zeros((8,8,3), dtype=np.uint8)
    cascade = demo.ActionCascade(tmp_path / "records", sink)
    first = cascade.fire("fire_colors", "HIGH", reply(), frame)
    before = sink.read_bytes()
    assert first["test_sink"] == "written"
    assert json.loads(before)["kind"] == "non_emergency_test"
    second = cascade.fire("person_in_zone", "MEDIUM", reply(event="person_in_zone"), frame)
    assert second["test_sink"] == "not_attempted" and sink.read_bytes() == before
    restarted = demo.ActionCascade(tmp_path / "records", sink)
    result = restarted.fire("fire_colors", "HIGH", reply(), frame)
    assert result["test_sink"] == "unknown_no_retry" and sink.read_bytes() == before


def test_provider_error_becomes_refusal(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("test timeout")
    monkeypatch.setattr(demo.urllib.request, "urlopen", fail)
    frame = np.zeros((8,8,3), dtype=np.uint8)
    raw = demo.ask_vlm(frame, "test")
    sink = tmp_path / "sink"
    event = demo.ActionCascade(tmp_path / "records", sink).fire("fire_colors", "HIGH", raw, frame)
    assert event["review"] == "refused" and not sink.exists()


@pytest.mark.parametrize("sink", ["https://example.invalid/sink", "//server/share/sink"])
def test_remote_sink_refused(tmp_path, sink):
    with pytest.raises(ValueError, match="local file"):
        demo.ActionCascade(tmp_path, sink)
