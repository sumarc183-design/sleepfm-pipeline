import pytest

from src.data.nsrr_annotations import reference_ahi_nsrr
from src.data.nsrr_channels import channel_plan
from src.data.ucddb_annotations import edf_start_seconds, parse_ucddb

HEADER = """                              Respiratory Event List
        Respiratory Event           Desaturation   Snore Arousal     B/T
 Time       Type   PB/CS  Duration  Low    %Drop                 Rate  Change
"""
UCDDB_LABELS = ["Lefteye", "RightEye", "EMG", "C3A2", "C4A1", "ECG", "SpO2", "Sound", "Flow", "Sum",
                "ribcage", "abdo", "BodyPos", "Pulse"]  # fmt: skip


def write(tmp_path, stages, events):
    stage, resp = tmp_path / "s_stage.txt", tmp_path / "s_respevt.txt"
    stage.write_text("\n".join(str(c) for c in stages) + "\n")
    resp.write_text(HEADER + "".join(events))
    return stage, resp


def test_parse_and_reference(tmp_path):
    stages = [0] * 20 + [3] * 200 + [5] * 20 + [1] * 40 + [6, 7]  # 2 h 20 de sommeil noté
    events = [
        "00:29:13  HYP-C             16       89.9    4.1     -     -      64.7   -5.7 \n",
        "00:40:00  APNEA-O           12       90      3       -     -              \n",
        "00:50:00  HYP-O    PB       20       92      2.1     -     -              \n",
    ]
    stage, resp = write(tmp_path, stages, events)
    parsed = parse_ucddb(stage, resp, start_s=11 * 60 + 4)  # EDF commencé à 00:11:04
    assert parsed["events"][0].tolist() == [1089.0, 1105.0, 3.0]  # 18 min 09 s après le début
    assert parsed["unscored_stage_events"] == 2 and parsed["recording_end_s"] == 282 * 30
    assert parsed["other_concepts"] == {"periodic_breathing_flag": 1}
    reference = reference_ahi_nsrr(parsed)
    assert reference["sleep_h"] == pytest.approx(260 * 30 / 3600)
    assert reference["event_counts"]["resp_hypopnea"] == 2


def test_events_after_midnight_wrap(tmp_path):
    stage, resp = write(tmp_path, [2] * 10, ["00:05:00  APNEA-C           10       90      3\n"])
    parsed = parse_ucddb(stage, resp, start_s=23 * 3600 + 55 * 60)  # début 23:55
    assert parsed["events"][0, 0] == 600.0


def test_unknown_type_or_code_blocks_reference(tmp_path):
    stage, resp = write(tmp_path, [2] * 200 + [9], ["00:20:00  CS-APNEA          10       90      3\n"])
    parsed = parse_ucddb(stage, resp, start_s=0)
    assert parsed["unknown_concepts"] == {"stage 9": 1, "CS-APNEA": 1}
    with pytest.raises(ValueError):
        reference_ahi_nsrr(parsed)


def test_edf_start_and_channel_plan(tmp_path):
    edf = tmp_path / "x.rec"
    edf.write_bytes(b" " * 168 + b"01.01.02" + b"00.11.04" + b" " * 72)
    assert edf_start_seconds(edf) == 664
    plan = channel_plan(UCDDB_LABELS, "ucddb", "sleepfm")
    assert plan["indices"]["C3-M2"] == 3 and plan["indices"]["AIRFLOW"] == 8
    assert plan["zero_filled"] == ["F3-M2", "F4-M1", "O1-M2", "O2-M1"]
    assert channel_plan(UCDDB_LABELS, "ucddb", "osf")["lost_vs_physionet"] == []
