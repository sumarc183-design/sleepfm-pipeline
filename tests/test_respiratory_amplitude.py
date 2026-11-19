"""Résumé d'amplitude : ratio indéfini publié en null, JSON strict (audit A6 du 19-11)."""

import json

from scripts import respiratory_amplitude_physionet2018 as amplitude


def night(scale):
    return {"ABD": {"robust_scale": scale, "median": 0.0}}


def test_flat_channel_gives_null_ratio_with_reason():
    flat = amplitude.summary({f"n{i}": night(0.0 if i < 2 else 1.0 + i) for i in range(10)}, ("ABD",))
    assert flat["ABD"]["p95_over_p5"] is None and "plate" in flat["ABD"]["p95_over_p5_undefined"]
    json.dumps(flat, allow_nan=False)  # lisible par un parseur strict
    normal = amplitude.summary({f"n{i}": night(1.0 + i) for i in range(10)}, ("ABD",))
    assert normal["ABD"]["p95_over_p5"] > 1 and normal["ABD"]["p95_over_p5_undefined"] is None
