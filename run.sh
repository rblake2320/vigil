#!/usr/bin/env bash
# run.sh — Vigil launcher shortcut
# Usage: ./run.sh [command] [args]
#
# Commands:
#   coach          Full screen coaching (Cosmos + TTS)
#   coach-proc     Coach against a procedure
#   wifi-sim       WiFi sim piped into coach
#   wifi-csi       Live ESP32 CSI piped into coach
#   describe       Screen narration only
#   source         Raw perception signals (no coaching)
#
# Examples:
#   ./run.sh coach
#   ./run.sh coach-proc watcher_procedures/it_basic.json
#   ./run.sh wifi-sim

PYTHON=~/miniconda3/bin/python
DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-coach}" in
  coach)
    exec $PYTHON -u "$DIR/perception/coach.py" --describe "${@:2}"
    ;;
  coach-proc)
    exec $PYTHON -u "$DIR/perception/coach.py" --procedure "${2:?Usage: ./run.sh coach-proc <procedure.json>}" "${@:3}"
    ;;
  wifi-sim)
    $PYTHON -u "$DIR/perception/wifi_source.py" --mode sim 2>/dev/null | \
    exec $PYTHON -u "$DIR/perception/coach.py" --describe --stdin "${@:2}"
    ;;
  wifi-csi)
    $PYTHON -u "$DIR/perception/wifi_source.py" --mode csi 2>/dev/null | \
    exec $PYTHON -u "$DIR/perception/coach.py" --describe --stdin "${@:2}"
    ;;
  wifi-rssi)
    $PYTHON -u "$DIR/perception/wifi_source.py" --mode rssi --iface "${2:-wlan0}" 2>/dev/null | \
    exec $PYTHON -u "$DIR/perception/coach.py" --describe --stdin "${@:3}"
    ;;
  describe)
    exec $PYTHON -u "$DIR/perception/coach.py" --describe --no-cosmos "${@:2}"
    ;;
  source)
    exec $PYTHON -u "$DIR/perception/spark2_source.py" "${@:2}"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Commands: coach, coach-proc, wifi-sim, wifi-csi, wifi-rssi, describe, source"
    exit 1
    ;;
esac
