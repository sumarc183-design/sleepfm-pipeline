import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.embed_nsrr import subject_list
from scripts.freeze_nsrr_subjects import freeze

ROOT = Path(__file__).resolve().parents[1]


def touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


def test_freeze_keeps_pairs_and_reports_the_rest(tmp_path):
    edf, xml = tmp_path / "edfs", tmp_path / "xml"
    touch(edf, "mesa-sleep-0002.edf", "mesa-sleep-0001.edf", "mesa-sleep-0003.edf", "notes.edf", "readme.txt")
    touch(xml, "mesa-sleep-0001-nsrr.xml", "mesa-sleep-0002-nsrr.xml", "mesa-sleep-0009-nsrr.xml",
          "mesa-sleep-0001-profusion.xml")  # fmt: skip
    payload = freeze("mesa", edf, xml)
    assert payload["subjects"] == ["mesa-sleep-0001", "mesa-sleep-0002"]
    assert payload["edf_without_xml"] == ["mesa-sleep-0003"]
    assert payload["xml_without_edf"] == ["mesa-sleep-0009"]
    assert payload["ignored_files"] == ["notes.edf"]


def test_shhs_second_visit_excluded(tmp_path):
    edf, xml = tmp_path / "edfs", tmp_path / "xml"
    touch(edf, "shhs1-200001.edf", "shhs2-200001.edf")
    touch(xml, "shhs1-200001-nsrr.xml", "shhs2-200001-nsrr.xml")
    payload = freeze("shhs1", edf, xml)
    assert payload["subjects"] == ["shhs1-200001"]
    assert payload["ignored_files"] == ["shhs2-200001.edf", "shhs2-200001-nsrr.xml"]


def test_cli_writes_list_read_by_embed_and_never_overwrites(tmp_path):
    edf, xml, out = tmp_path / "edfs", tmp_path / "xml", tmp_path / "list.json"
    touch(edf, "mesa-sleep-0001.edf")
    touch(xml, "mesa-sleep-0001-nsrr.xml")
    command = [sys.executable, "scripts/freeze_nsrr_subjects.py", "--cohort", "mesa", "--edf-dir", str(edf),
               "--annotations-dir", str(xml), "--out", str(out)]  # fmt: skip
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    subjects, digest = subject_list(out, "mesa")
    assert subjects == ["mesa-sleep-0001"] and len(digest) == 64
    before = out.read_bytes()
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0 and out.read_bytes() == before
    assert json.loads(before)["rule"].startswith("EDF et annotations")


def test_empty_selection_refused(tmp_path):
    touch(tmp_path / "edfs", "mesa-sleep-0001.edf")
    touch(tmp_path / "xml")
    with pytest.raises(SystemExit):
        freeze("mesa", tmp_path / "edfs", tmp_path / "xml")


def test_edf_names_from_index_file(tmp_path):
    touch(tmp_path / "xml", "mesa-sleep-0001-nsrr.xml", "mesa-sleep-0002-nsrr.xml")
    index = tmp_path / "edfs.txt"
    index.write_text("polysomnography/edfs/mesa-sleep-0001.edf\n\nmesa-sleep-0002.edf\nmesa-sleep-0001.edf\n")
    assert freeze("mesa", index, tmp_path / "xml")["subjects"] == ["mesa-sleep-0001", "mesa-sleep-0002"]
