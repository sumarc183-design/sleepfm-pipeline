import pytest

from src.data.nsrr_annotations import reference_ahi_nsrr
from src.data.nsrr_channels import channel_plan
from src.data.nsrr_signals import signal_names
from src.data.psg_audio_annotations import parse_rml

NS = "http://www.respironics.com/PatientStudy.xsd"


def rml(stages, events, duration=7800):
    stage_xml = "".join(f'<Stage Type="{t}" Start="{s}"/>' for t, s in stages)
    event_xml = "".join(f'<Event Family="{f}" Type="{t}" Start="{s}" Duration="{d}"/>' for f, t, s, d in events)
    return (
        "﻿<?xml version='1.0' encoding='utf-8'?>"
        f'<PatientStudy xmlns="{NS}"><Acquisition><Sessions><Session><RecordingStart>x</RecordingStart>'
        f"<Duration>{duration}</Duration></Session></Sessions></Acquisition><ScoringData><Events>{event_xml}</Events>"
        "<StagingData><UserStaging><NeuroAdultAASMStaging>"
        f"{stage_xml}</NeuroAdultAASMStaging></UserStaging><MachineStaging><NeuroAdultAASMStaging>"
        '<Stage Type="QuietSleep" Start="0"/></NeuroAdultAASMStaging></MachineStaging></StagingData>'
        "</ScoringData></PatientStudy>"
    ).encode()


def test_parse_reference_and_non_target():
    events = [
        ("Respiratory", "Hypopnea", 700, 20),
        ("Respiratory", "ObstructiveApnea", 800.5, 12),
        ("Respiratory", "PeriodicRespiration", 900, 60),
        ("SpO2", "RelativeDesaturation", 720, 10),
    ]
    parsed = parse_rml(rml([("Wake", 0), ("NonREM2", 600), ("REM", 6000)], events))
    assert [s[2] for s in parsed["stages"]] == ["W", "N2", "R"] and parsed["stages"][-1][1] == 1800
    assert parsed["unknown_concepts"] == {} and parsed["recording_end_s"] == 7800
    assert parsed["other_concepts"] == {"PeriodicRespiration": 1, "SpO2:RelativeDesaturation": 1}
    reference = reference_ahi_nsrr(parsed)
    assert reference["sleep_h"] == pytest.approx(7200 / 3600) and reference["events"] == 2


def test_unknown_type_blocks_and_bad_xml_rejected():
    parsed = parse_rml(rml([("NonREM2", 0)], [("Respiratory", "SomethingApnea", 10, 10)]))
    with pytest.raises(ValueError):
        reference_ahi_nsrr(parsed)
    with pytest.raises(ValueError, match="RML illisible"):
        parse_rml(b"<PatientStudy><broken")


def test_flow_thermistor_selected_by_transducer():
    labels = ["EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EMG Chin", "ECG I", "Flow Patient", "Flow Patient",
              "Effort THO", "Effort ABD", "SpO2"]  # fmt: skip
    transducers = [""] * 5 + ["Thermistor", "Pressure"] + [""] * 3
    names = signal_names(labels, transducers)
    assert names[5:7] == ["Flow Patient|Thermistor", "Flow Patient|Pressure"] and names[0] == "EEG C3-A2"
    plan = channel_plan(names, "psgaudio", "sleepfm")
    assert plan["indices"]["AIRFLOW"] == 5 and plan["zero_filled"] == ["F3-M2", "F4-M1", "O1-M2", "O2-M1"]
