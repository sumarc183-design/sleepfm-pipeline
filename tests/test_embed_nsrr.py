import json

import pytest
import torch

from scripts.embed_nsrr import embed_subject, identity_scale, subject_list
from src.models.osf_vit1d import vit_base
from tests.test_nsrr_signals import SHHS, synthetic, write_edf


class FakeSleepFM:
    def embed_combined(self, epochs):
        assert epochs.shape[1:] == (13, 7680)
        return torch.ones(len(epochs), 1536)


@pytest.fixture
def edf(tmp_path):
    path = tmp_path / "shhs1-200001.edf"
    write_edf(path, SHHS, [synthetic(c[2], c[1]) / (100 if c[1] in ("", "mV") else 1) for c in SHHS])
    return path


def test_embed_subject_both_models(edf):
    torch.manual_seed(0)
    out = embed_subject(edf, "shhs1", FakeSleepFM(), vit_base(num_leads=12).eval(), identity_scale, 2, "cpu")
    assert out["X_sleepfm"].shape == (3, 1536) and out["X_osf"].shape == (3, 768)
    assert out["onset_s"].tolist() == [0.0, 30.0, 60.0] and out["info"]["osf_flat"] == []


def test_embed_subject_sleepfm_needs_rule(edf):
    with pytest.raises(ValueError):
        embed_subject(edf, "shhs1", FakeSleepFM(), None, None, 2, "cpu")


def test_subject_list(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps({"cohort": "mesa", "subjects": ["mesa-sleep-0001"]}))
    subjects, sha = subject_list(path, "mesa")
    assert subjects == ["mesa-sleep-0001"] and len(sha) == 64
    with pytest.raises(SystemExit):
        subject_list(path, "shhs1")
    path.write_text(json.dumps({"cohort": "mesa", "subjects": ["a", "a"]}))
    with pytest.raises(SystemExit):
        subject_list(path, "mesa")


def test_embed_subject_with_validated_rule(edf):
    from scripts.embed_nsrr import RESPIRATORY_RULES

    out = embed_subject(edf, "shhs1", FakeSleepFM(), None, RESPIRATORY_RULES["physionet_median_v1"], 2, "cpu")
    assert out["X_sleepfm"].shape == (3, 1536) and "X_osf" not in out
