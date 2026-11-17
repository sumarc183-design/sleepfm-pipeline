"""Page de relecture : gains WFDB, enveloppes, annotations, bords de nuit, aveugle sur les modèles."""

import json

import numpy as np

from scripts import build_hypopnea_review as page


def test_channel_gains_reads_gain_and_baseline(tmp_path):
    hea = tmp_path / "x.hea"
    hea.write_text(
        "x 2 200 1000\nx.mat 16+24 1/uV 16 0 5 0 0 AIRFLOW\nx.mat 16+24 655.35(-32768)/% 16 0 0 0 0 SaO2\n",
        encoding="utf-8",
    )
    assert page.channel_gains(hea) == {"AIRFLOW": (1.0, 0.0), "SaO2": (655.35, -32768.0)}


def test_envelope_keeps_min_and_max_of_each_column():
    signal = np.array([0, 5, -1, 2, 9, 3], dtype=float)
    assert page.envelope(signal, columns=3) == [0.0, 5.0, -1.0, 2.0, 3.0, 9.0]


def test_annotation_intervals_keeps_overlapping_events_in_relative_seconds():
    samples = [100, 300, 1000, 1400, 5000, 5200]
    notes = [
        "(resp_hypopnea",
        "resp_hypopnea)",
        "(arousal_rera",
        "arousal_rera)",
        "(resp_centralapnea",
        "resp_centralapnea)",
    ]
    events = page.annotation_intervals(samples, notes, rate=100, start_s=2.0, end_s=20.0)
    assert events == [
        {"name": "resp_hypopnea", "start": -1.0, "end": 1.0},
        {"name": "arousal_rera", "start": 8.0, "end": 12.0},
    ]


def test_window_payload_is_clipped_at_end_of_night():
    rate, n = 10, 10 * 600
    signals = {name: np.arange(n, dtype=float) for name, _ in page.CHANNELS}
    payload = page.window_payload(signals, rate, onset_s=590.0, events=[])
    assert payload["start"] == 480.0 and payload["box"] == [110.0, 140.0]
    assert all(len(v) == 2 * page.COLUMNS for v in payload["traces"].values())


def test_render_hides_groups_and_model_outputs():
    window = {"id": "F01", "box": [45.0, 75.0], "traces": {k: [0.0, 1.0] for k, _ in page.CHANNELS}, "events": []}
    text = page.render([window], "abc")
    data = json.loads(text.split('type="application/json">')[1].split("</script>")[0])
    assert data["windows"] == [window]
    for leaked in ("probability", "group", "osf", "sleepfm", "tr0"):
        assert leaked not in json.dumps(data).lower()
