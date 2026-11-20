"""Fusion des parties Kaggle : manifeste commun, doublons identiques seulement, chemins à plat."""

import json
import zipfile

import pytest

from scripts import merge_embedding_parts as merge

MANIFEST = {"cohort": "mesa", "subjects_sha256": "abc"}
SUBJECTS = json.loads((merge.ROOT / "configs/nsrr_mesa_subjects.json").read_text(encoding="utf-8"))["subjects"]


def part(tmp_path, name, files, manifest=MANIFEST):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        if manifest is not None:
            archive.writestr("manifest.json", json.dumps(manifest))
        for file_name, data in files.items():
            archive.writestr(file_name, data)
    return path


def night(subject, data=b"npz"):
    return {f"{subject}.npz": data, f"{subject}.json": b"{}"}


def test_parts_are_merged_and_coverage_reported(tmp_path):
    a = part(tmp_path, "p0.zip", night(SUBJECTS[0]) | {"run_status.json": b"{}"})
    b = part(tmp_path, "p1.zip", night(SUBJECTS[1]) | {f"{SUBJECTS[2]}.skipped": b"EDF tronque"})
    report = merge.merge([a, b], tmp_path / "out")
    assert report["done"] == 2 and report["rejected"] == 1
    assert len(report["missing"]) == len(SUBJECTS) - 3
    assert not (tmp_path / "out" / "run_status.json").exists()  # recalculé ensuite par embed_nsrr
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == MANIFEST


@pytest.mark.parametrize(
    ("second", "match"),
    [
        ({"files": night(SUBJECTS[1]), "manifest": MANIFEST | {"subjects_sha256": "autre"}}, "manifeste"),
        ({"files": night(SUBJECTS[0], b"autre calcul")}, "différent entre"),
        ({"files": {"../evil.npz": b"x"}}, "chemins refusés"),
        ({"files": night(SUBJECTS[1]), "manifest": None}, "manifest.json absent"),
        ({"files": {"notes.txt": b"x"}}, "inattendu"),
    ],
)
def test_inconsistent_parts_are_refused_before_writing(tmp_path, second, match):
    a = part(tmp_path, "p0.zip", night(SUBJECTS[0]))
    b = part(tmp_path, "p1.zip", second["files"], second.get("manifest", MANIFEST))
    with pytest.raises(ValueError, match=match):
        merge.merge([a, b], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_identical_duplicate_is_accepted_but_local_conflict_refused(tmp_path):
    a = part(tmp_path, "p0.zip", night(SUBJECTS[0]))
    b = part(tmp_path, "p1.zip", night(SUBJECTS[0]))
    assert merge.merge([a, b], tmp_path / "out")["done"] == 1
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / f"{SUBJECTS[0]}.npz").write_bytes(b"calcul CPU")
    with pytest.raises(ValueError, match="déjà présents et différents"):
        merge.merge([a], tmp_path / "cpu")
