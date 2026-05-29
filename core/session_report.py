"""
session_report.py — CheatVision Session Report Generator
=========================================================
Generates a self-contained HTML report at the end of a CheatVision session.
No external dependencies — stdlib only.

Usage:
    report = SessionReport("sess-abc123", "CS2")
    report.record_event("CLEAN", 0.12)
    report.record_event("SUSPICIOUS", 0.55)
    report.add_evidence({"ts": 42.0, "verdict": "CONFIRMED_CHEAT",
                         "score": 0.91, "sha256": "abc...", "path": "/tmp/clip.mp4"})
    path = report.generate()          # saves HTML, returns path
    print(path)

Flask integration:
    app.add_url_rule("/report", "report", make_report_route(report))
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Verdict constants (mirrors analyzer.Verdict so we avoid circular imports)
# ---------------------------------------------------------------------------
_VERDICT_ORDER = ["CLEAN", "SUSPICIOUS", "LIKELY_CHEAT", "CONFIRMED_CHEAT"]
_VERDICT_COLOR = {
    "CLEAN":           "#00ff88",   # accent green
    "SUSPICIOUS":      "#ffd700",   # yellow
    "LIKELY_CHEAT":    "#ff8c00",   # orange
    "CONFIRMED_CHEAT": "#ff2244",   # red
}
_VERDICT_BG = {
    "CLEAN":           "rgba(0,255,136,0.15)",
    "SUSPICIOUS":      "rgba(255,215,0,0.15)",
    "LIKELY_CHEAT":    "rgba(255,140,0,0.20)",
    "CONFIRMED_CHEAT": "rgba(255,34,68,0.25)",
}


# ---------------------------------------------------------------------------
# SessionReport
# ---------------------------------------------------------------------------
class SessionReport:
    """
    Accumulates session data and generates a professional HTML report.

    Parameters
    ----------
    session_id : str
        Unique identifier for the session (e.g. UUID or slug).
    game : str
        Human-readable game title (e.g. "CS2", "Valorant").
    """

    def __init__(self, session_id: str, game: str) -> None:
        self.session_id: str = session_id
        self.game: str = game
        self.start_time: float = time.time()
        self.events: List[Tuple[float, str, float]] = []   # (ts, verdict, score)
        self.evidence_clips: List[dict] = []
        self.peak_score: float = 0.0
        self.frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_event(
        self,
        verdict: str,
        score: float,
        ts: Optional[float] = None,
        signals: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a verdict event (call once per analysis cycle / second).

        Parameters
        ----------
        verdict  : verdict string — one of CLEAN / SUSPICIOUS / LIKELY_CHEAT / CONFIRMED_CHEAT
        score    : cheat probability 0.0–1.0
        ts       : absolute timestamp; defaults to now
        signals  : optional dict of signal_name → probability (for Signal Breakdown)
        """
        if ts is None:
            ts = time.time()
        score = max(0.0, min(1.0, float(score)))
        self.events.append((ts, str(verdict), score, signals or {}))
        if score > self.peak_score:
            self.peak_score = score

    def add_evidence(self, clip_dict: dict) -> None:
        """
        Add an evidence clip entry (as produced by AutoEvidence or similar).

        Expected keys (all optional except *ts*):
            ts       : float  — session-relative or absolute timestamp
            verdict  : str
            score    : float
            sha256   : str    — hex digest of the clip file
            path     : str    — filesystem path to the clip
            note     : str    — free-form annotation
        """
        self.evidence_clips.append(dict(clip_dict))

    def generate(self, output_path: Optional[str] = None) -> str:
        """
        Render the HTML report, write it to disk, and return the file path.

        Parameters
        ----------
        output_path : str, optional
            Explicit output file path.  If omitted, a name is auto-generated
            in the current working directory as
            ``cheatvision_<session_id>_<date>.html``.
        """
        if output_path is None:
            date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_sid = "".join(c if c.isalnum() or c in "-_" else "_"
                               for c in self.session_id)[:32]
            output_path = os.path.join(
                os.getcwd(),
                f"cheatvision_{safe_sid}_{date_tag}.html",
            )

        html_content = self._render_html()
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _duration_secs(self) -> float:
        if not self.events:
            return time.time() - self.start_time
        return self.events[-1][0] - self.start_time

    def _fmt_duration(self, secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def _confirmed_count(self) -> int:
        return sum(1 for e in self.events if e[1] == "CONFIRMED_CHEAT")

    def _signal_breakdown(self) -> Dict[str, float]:
        """Aggregate signal probabilities across all recorded events."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for e in self.events:
            signals = e[3] if len(e) > 3 else {}
            for name, prob in signals.items():
                totals[name] = totals.get(name, 0.0) + float(prob)
                counts[name] = counts.get(name, 0) + 1
        if not totals:
            return {}
        avgs = {k: totals[k] / counts[k] for k in totals}
        return dict(sorted(avgs.items(), key=lambda x: x[1], reverse=True))

    def _chain_integrity(self) -> str:
        """Verify SHA-256 hashes for all evidence clips that have them."""
        broken = 0
        verified = 0
        for clip in self.evidence_clips:
            path = clip.get("path", "")
            expected = clip.get("sha256", "")
            if not path or not expected:
                continue
            try:
                sha = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        sha.update(chunk)
                actual = sha.hexdigest()
                if actual.lower() == expected.lower():
                    verified += 1
                else:
                    broken += 1
            except OSError:
                broken += 1
        if broken:
            return f"broken ({broken} file(s) failed)"
        if verified:
            return f"verified ({verified} file(s))"
        return "no clips to verify"

    # ------------------------------------------------------------------
    # Timeline bar-chart (pure HTML/CSS, no JS)
    # ------------------------------------------------------------------

    def _render_timeline(self) -> str:
        if not self.events:
            return "<p class='muted'>No events recorded.</p>"

        duration = max(self._duration_secs(), 1.0)
        # One bar per event
        bars_html = []
        for ev in self.events:
            ts, verdict, score, _ = ev[0], ev[1], ev[2], ev[3] if len(ev) > 3 else {}
            rel = ts - self.start_time
            left_pct = (rel / duration) * 100.0
            height_pct = score * 100.0
            color = _VERDICT_COLOR.get(verdict, "#888")
            tip = f"{verdict} | score={score:.3f} | t={self._fmt_duration(rel)}"
            bars_html.append(
                f'<div class="tl-bar" style="left:{left_pct:.3f}%;'
                f'height:{height_pct:.1f}%;background:{color};" title="{html.escape(tip)}"></div>'
            )

        # X-axis tick marks every minute (or every 10 s if session < 2 min)
        tick_interval = 60 if duration >= 120 else 10
        ticks_html = []
        t = 0
        while t <= duration:
            pct = (t / duration) * 100.0
            label = f"{int(t//60)}m" if t % 60 == 0 else f"{int(t)}s"
            ticks_html.append(
                f'<div class="tl-tick" style="left:{pct:.2f}%">'
                f'<span class="tl-tick-label">{label}</span></div>'
            )
            t += tick_interval
        if t - tick_interval < duration:
            pct = 100.0
            label = self._fmt_duration(duration)
            ticks_html.append(
                f'<div class="tl-tick" style="left:100%">'
                f'<span class="tl-tick-label">{html.escape(label)}</span></div>'
            )

        # Y-axis labels (0.0, 0.25, 0.5, 0.75, 1.0)
        y_labels = []
        for v in [1.0, 0.75, 0.5, 0.25, 0.0]:
            bottom = v * 100
            y_labels.append(
                f'<div class="tl-ylabel" style="bottom:{bottom:.0f}%">{v:.2f}</div>'
            )

        # Threshold lines
        thresholds = [
            (0.40, "SUSPICIOUS",      _VERDICT_COLOR["SUSPICIOUS"]),
            (0.65, "LIKELY_CHEAT",    _VERDICT_COLOR["LIKELY_CHEAT"]),
            (0.82, "CONFIRMED_CHEAT", _VERDICT_COLOR["CONFIRMED_CHEAT"]),
        ]
        thresh_lines = []
        for thresh, label, color in thresholds:
            bottom = thresh * 100
            thresh_lines.append(
                f'<div class="tl-threshold" style="bottom:{bottom:.1f}%;border-color:{color};">'
                f'<span class="tl-thresh-label" style="color:{color};">'
                f'{label} ({thresh})</span></div>'
            )

        return f"""
<div class="timeline-container">
  <div class="tl-ylabel-axis">{''.join(y_labels)}</div>
  <div class="tl-chart">
    {''.join(thresh_lines)}
    {''.join(bars_html)}
    <div class="tl-ticks">{''.join(ticks_html)}</div>
  </div>
</div>
"""

    # ------------------------------------------------------------------
    # Signal breakdown section
    # ------------------------------------------------------------------

    def _render_signals(self) -> str:
        signals = self._signal_breakdown()
        if not signals:
            # Try to derive from verdict distribution as fallback
            verdict_counts = Counter(e[1] for e in self.events)
            if not verdict_counts:
                return "<p class='muted'>No signal data recorded.</p>"
            rows = []
            for verdict in _VERDICT_ORDER:
                count = verdict_counts.get(verdict, 0)
                if count == 0:
                    continue
                pct = (count / len(self.events)) * 100
                color = _VERDICT_COLOR.get(verdict, "#888")
                rows.append(f"""
  <div class="signal-row">
    <span class="signal-name">{html.escape(verdict)}</span>
    <div class="signal-bar-wrap">
      <div class="signal-bar" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
    <span class="signal-pct">{count} events ({pct:.1f}%)</span>
  </div>""")
            return '<div class="signal-list">' + "".join(rows) + "</div>"

        rows = []
        for name, avg in list(signals.items())[:12]:   # top 12 signals
            pct = avg * 100
            color = (
                _VERDICT_COLOR["CONFIRMED_CHEAT"] if avg >= 0.82 else
                _VERDICT_COLOR["LIKELY_CHEAT"]    if avg >= 0.65 else
                _VERDICT_COLOR["SUSPICIOUS"]       if avg >= 0.40 else
                _VERDICT_COLOR["CLEAN"]
            )
            rows.append(f"""
  <div class="signal-row">
    <span class="signal-name">{html.escape(name)}</span>
    <div class="signal-bar-wrap">
      <div class="signal-bar" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
    <span class="signal-pct">{avg:.3f}</span>
  </div>""")
        return '<div class="signal-list">' + "".join(rows) + "</div>"

    # ------------------------------------------------------------------
    # Evidence clips section
    # ------------------------------------------------------------------

    def _render_evidence(self) -> str:
        if not self.evidence_clips:
            return "<p class='muted'>No evidence clips saved this session.</p>"

        rows = []
        for i, clip in enumerate(self.evidence_clips, 1):
            ts_raw = clip.get("ts", 0.0)
            # Determine if ts is absolute or relative
            if isinstance(ts_raw, float) and ts_raw > 1e9:
                rel = ts_raw - self.start_time
            else:
                rel = float(ts_raw)
            rel_fmt = self._fmt_duration(max(0.0, rel))
            verdict = clip.get("verdict", "UNKNOWN")
            score = float(clip.get("score", 0.0))
            sha256 = clip.get("sha256", "—")
            path = clip.get("path", "—")
            note = clip.get("note", "")
            color = _VERDICT_COLOR.get(verdict, "#888")
            bg = _VERDICT_BG.get(verdict, "rgba(255,255,255,0.05)")

            sha_display = sha256[:16] + "…" + sha256[-8:] if len(sha256) > 24 else sha256

            rows.append(f"""
<div class="evidence-card" style="border-left:3px solid {color};background:{bg};">
  <div class="evidence-header">
    <span class="evidence-idx">#{i}</span>
    <span class="evidence-verdict" style="color:{color};">{html.escape(verdict)}</span>
    <span class="evidence-score">score: {score:.3f}</span>
    <span class="evidence-time">@ {html.escape(rel_fmt)}</span>
  </div>
  <div class="evidence-detail">
    <div><span class="label">Path:</span>
         <span class="mono">{html.escape(str(path))}</span></div>
    <div><span class="label">SHA-256:</span>
         <span class="mono sha">{html.escape(sha_display)}</span></div>
    {f'<div><span class="label">Note:</span> {html.escape(note)}</div>' if note else ''}
  </div>
</div>""")

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Summary stats row
    # ------------------------------------------------------------------

    def _render_stats(self) -> str:
        total_events = len(self.events)
        confirmed = self._confirmed_count()
        clips = len(self.evidence_clips)
        verdict_counts = Counter(e[1] for e in self.events)

        dominant = max(verdict_counts, key=verdict_counts.get) if verdict_counts else "CLEAN"
        dom_color = _VERDICT_COLOR.get(dominant, "#888")

        def stat_card(label: str, value: str, color: str = "#00ff88", sub: str = "") -> str:
            return f"""
<div class="stat-card">
  <div class="stat-value" style="color:{color};">{html.escape(str(value))}</div>
  <div class="stat-label">{html.escape(label)}</div>
  {f'<div class="stat-sub">{html.escape(sub)}</div>' if sub else ''}
</div>"""

        cards = [
            stat_card("Total Events", str(total_events)),
            stat_card("CONFIRMED_CHEAT", str(confirmed),
                      color=_VERDICT_COLOR["CONFIRMED_CHEAT"] if confirmed else "#00ff88"),
            stat_card("Peak Score", f"{self.peak_score:.3f}",
                      color=self._score_color(self.peak_score)),
            stat_card("Evidence Clips", str(clips), color="#00aaff"),
            stat_card("Dominant Verdict", dominant, color=dom_color,
                      sub=f"{verdict_counts.get(dominant,0)} events"),
            stat_card("Frames Analyzed", str(self.frame_count) if self.frame_count else "—"),
        ]
        return '<div class="stats-row">' + "".join(cards) + "</div>"

    def _score_color(self, score: float) -> str:
        if score >= 0.82:
            return _VERDICT_COLOR["CONFIRMED_CHEAT"]
        if score >= 0.65:
            return _VERDICT_COLOR["LIKELY_CHEAT"]
        if score >= 0.40:
            return _VERDICT_COLOR["SUSPICIOUS"]
        return _VERDICT_COLOR["CLEAN"]

    # ------------------------------------------------------------------
    # Full HTML render
    # ------------------------------------------------------------------

    def _render_html(self) -> str:
        now_dt = datetime.now(tz=timezone.utc)
        start_dt = datetime.fromtimestamp(self.start_time, tz=timezone.utc)
        duration = self._duration_secs()
        chain_status = self._chain_integrity()
        chain_color = "#00ff88" if "verified" in chain_status or "no clips" in chain_status else "#ff2244"

        # Verdicts for per-verdict event counts (legend)
        verdict_counts = Counter(e[1] for e in self.events)
        legend_items = []
        for v in _VERDICT_ORDER:
            cnt = verdict_counts.get(v, 0)
            if cnt:
                c = _VERDICT_COLOR[v]
                legend_items.append(
                    f'<span class="legend-dot" style="background:{c};"></span>'
                    f'<span class="legend-label">{v} ({cnt})</span>'
                )

        css = """
:root {
  --bg:       #0a0a0a;
  --surface:  #111318;
  --surface2: #181b22;
  --border:   #1e2230;
  --accent:   #00ff88;
  --text:     #e0e6f0;
  --muted:    #5a6070;
  --font:     'Courier New', 'Consolas', 'Lucida Console', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  line-height: 1.6;
  padding: 0;
}
a { color: var(--accent); text-decoration: none; }

/* ── Page wrapper ── */
.page { max-width: 1100px; margin: 0 auto; padding: 32px 24px 60px; }

/* ── Header ── */
.report-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 24px;
  margin-bottom: 32px;
}
.logo-row {
  display: flex; align-items: center; gap: 16px; margin-bottom: 8px;
}
.logo {
  font-size: 26px; font-weight: bold; letter-spacing: 2px;
  color: var(--accent); text-shadow: 0 0 18px rgba(0,255,136,0.5);
}
.logo-sub { font-size: 11px; color: var(--muted); letter-spacing: 4px; text-transform: uppercase; }
.header-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
  margin-top: 16px;
}
.meta-item { color: var(--muted); }
.meta-item span { color: var(--text); }

/* ── Section headings ── */
.section { margin-bottom: 36px; }
.section-title {
  font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--accent); border-bottom: 1px solid var(--border);
  padding-bottom: 6px; margin-bottom: 18px;
}

/* ── Stats row ── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 8px;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  text-align: center;
}
.stat-value { font-size: 22px; font-weight: bold; margin-bottom: 4px; }
.stat-label { font-size: 10px; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; }
.stat-sub   { font-size: 10px; color: var(--muted); margin-top: 4px; }

/* ── Timeline ── */
.timeline-container {
  display: flex; gap: 8px; height: 200px;
}
.tl-ylabel-axis {
  position: relative; width: 36px; flex-shrink: 0;
}
.tl-ylabel {
  position: absolute; right: 0;
  font-size: 10px; color: var(--muted);
  transform: translateY(50%);
}
.tl-chart {
  flex: 1; position: relative;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.tl-bar {
  position: absolute; bottom: 0;
  width: 3px; min-height: 2px;
  border-radius: 2px 2px 0 0;
  opacity: 0.85;
  transition: opacity .1s;
}
.tl-bar:hover { opacity: 1; z-index: 10; }
.tl-threshold {
  position: absolute; left: 0; right: 0;
  border-top: 1px dashed;
  opacity: 0.4;
}
.tl-thresh-label {
  position: absolute; right: 4px;
  font-size: 9px; letter-spacing: 1px;
  top: -13px;
}
.tl-ticks {
  position: absolute; bottom: 0; left: 0; right: 0; height: 0;
}
.tl-tick {
  position: absolute; bottom: 0;
  border-left: 1px solid var(--border);
  height: 200px;
}
.tl-tick-label {
  position: absolute; bottom: -18px; left: 2px;
  font-size: 9px; color: var(--muted); white-space: nowrap;
}

/* ── Legend ── */
.legend {
  display: flex; flex-wrap: wrap; gap: 16px;
  margin-top: 12px;
}
.legend-dot {
  display: inline-block;
  width: 10px; height: 10px; border-radius: 2px;
  vertical-align: middle; margin-right: 4px;
}
.legend-label { font-size: 11px; color: var(--muted); }

/* ── Evidence cards ── */
.evidence-card {
  background: var(--surface);
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 10px;
}
.evidence-header {
  display: flex; gap: 16px; align-items: baseline;
  margin-bottom: 8px;
}
.evidence-idx     { color: var(--muted); font-size: 11px; }
.evidence-verdict { font-weight: bold; font-size: 13px; }
.evidence-score   { color: var(--text); }
.evidence-time    { color: var(--muted); margin-left: auto; }
.evidence-detail  { display: grid; gap: 4px; }
.evidence-detail .label { color: var(--muted); margin-right: 8px; }
.mono  { font-family: var(--font); word-break: break-all; }
.sha   { color: var(--accent); font-size: 11px; }

/* ── Signal breakdown ── */
.signal-list { display: grid; gap: 8px; }
.signal-row  { display: flex; align-items: center; gap: 12px; }
.signal-name {
  width: 160px; flex-shrink: 0; color: var(--text);
  font-size: 11px; letter-spacing: .5px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.signal-bar-wrap {
  flex: 1; background: var(--surface2); border-radius: 3px; height: 14px;
}
.signal-bar { height: 100%; border-radius: 3px; min-width: 2px; }
.signal-pct { width: 60px; text-align: right; color: var(--muted); font-size: 11px; }

/* ── Footer ── */
.report-footer {
  border-top: 1px solid var(--border);
  padding-top: 18px;
  margin-top: 40px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.footer-brand { color: var(--muted); font-size: 11px; letter-spacing: 1px; }
.footer-chain { font-size: 11px; }

/* ── Misc ── */
.muted { color: var(--muted); }
"""

        body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CheatVision Report — {html.escape(self.session_id)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="page">

  <!-- ══ HEADER ══════════════════════════════════════════════════════════ -->
  <div class="report-header">
    <div class="logo-row">
      <div>
        <div class="logo">⬡ CHEATVISION</div>
        <div class="logo-sub">Anti-Cheat Session Report</div>
      </div>
    </div>
    <div class="header-meta">
      <div class="meta-item">Session ID&nbsp;&nbsp;<span>{html.escape(self.session_id)}</span></div>
      <div class="meta-item">Game&nbsp;&nbsp;<span>{html.escape(self.game)}</span></div>
      <div class="meta-item">Start&nbsp;&nbsp;<span>{html.escape(start_dt.strftime('%Y-%m-%d %H:%M:%S UTC'))}</span></div>
      <div class="meta-item">Duration&nbsp;&nbsp;<span>{html.escape(self._fmt_duration(duration))}</span></div>
      <div class="meta-item">Generated&nbsp;&nbsp;<span>{html.escape(now_dt.strftime('%Y-%m-%d %H:%M:%S UTC'))}</span></div>
    </div>
  </div>

  <!-- ══ SUMMARY STATS ════════════════════════════════════════════════ -->
  <div class="section">
    <div class="section-title">Summary</div>
    {self._render_stats()}
  </div>

  <!-- ══ VERDICT TIMELINE ═════════════════════════════════════════════ -->
  <div class="section">
    <div class="section-title">Verdict Timeline</div>
    {self._render_timeline()}
    <div class="legend">
      {''.join(legend_items) if legend_items else '<span class="muted">No events.</span>'}
    </div>
  </div>

  <!-- ══ EVIDENCE CLIPS ════════════════════════════════════════════════ -->
  <div class="section">
    <div class="section-title">Evidence Clips ({len(self.evidence_clips)})</div>
    {self._render_evidence()}
  </div>

  <!-- ══ SIGNAL BREAKDOWN ══════════════════════════════════════════════ -->
  <div class="section">
    <div class="section-title">Signal Breakdown</div>
    {self._render_signals()}
  </div>

  <!-- ══ FOOTER ════════════════════════════════════════════════════════ -->
  <div class="report-footer">
    <div class="footer-brand">Generated by CheatVision</div>
    <div class="footer-chain">
      SHA-256 chain:
      <span style="color:{chain_color};">{html.escape(chain_status)}</span>
    </div>
  </div>

</div><!-- /page -->
</body>
</html>"""

        return body


# ---------------------------------------------------------------------------
# Flask route factory
# ---------------------------------------------------------------------------

def make_report_route(report: SessionReport):
    """
    Return a callable suitable for registering as a Flask route.

    Example::

        app = Flask(__name__)
        app.add_url_rule("/report", "report", make_report_route(report))

    The handler generates the HTML on every request (so it reflects live data
    if the session is still in progress) and serves it inline.
    """

    def route_handler():
        # Late import — only needed when Flask is actually installed
        try:
            from flask import Response  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Flask is required for make_report_route. "
                "Install it with: pip install flask"
            )

        html_content = report._render_html()
        return Response(html_content, status=200, mimetype="text/html; charset=utf-8")

    route_handler.__name__ = f"cheatvision_report_{report.session_id}"
    return route_handler


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import sys

    rng = random.Random(42)

    r = SessionReport("demo-sess-001", "CS2")
    r.frame_count = 1800

    # Simulate a 5-minute session: 300 events at 1/sec
    base = r.start_time
    verdicts = ["CLEAN"] * 60 + ["SUSPICIOUS"] * 40 + ["LIKELY_CHEAT"] * 20 + ["CONFIRMED_CHEAT"] * 10
    rng.shuffle(verdicts)
    verdicts = verdicts[:300]
    score_map = {
        "CLEAN":           (0.05, 0.38),
        "SUSPICIOUS":      (0.40, 0.64),
        "LIKELY_CHEAT":    (0.65, 0.81),
        "CONFIRMED_CHEAT": (0.82, 0.99),
    }
    for i, v in enumerate(verdicts):
        lo, hi = score_map[v]
        sc = rng.uniform(lo, hi)
        signals = {
            "aim_snap_velocity": rng.uniform(0, 1),
            "path_linearity":    rng.uniform(0, 1),
            "preaim_score":      rng.uniform(0, 1),
            "wall_tracking":     rng.uniform(0, 1),
            "impossible_shots":  rng.uniform(0, 0.5),
        }
        r.record_event(v, sc, ts=base + i, signals=signals)

    r.add_evidence({
        "ts":      base + 142,
        "verdict": "CONFIRMED_CHEAT",
        "score":   0.94,
        "sha256":  "a3f4b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3",
        "path":    "/tmp/evidence_0142.mp4",
        "note":    "Snapped 180 degrees through wall 340ms before enemy became visible",
    })
    r.add_evidence({
        "ts":      base + 237,
        "verdict": "LIKELY_CHEAT",
        "score":   0.77,
        "sha256":  "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
        "path":    "/tmp/evidence_0237.mp4",
    })

    out = r.generate()
    print(f"Report written to: {out}")
    if "--open" in sys.argv:
        import subprocess
        subprocess.Popen(["xdg-open", out])
