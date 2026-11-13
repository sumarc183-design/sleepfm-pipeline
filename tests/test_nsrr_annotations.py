import numpy as np
import pytest

from src.data.nsrr_annotations import epoch_stages, parse_nsrr_xml, reference_ahi_nsrr


def scored(event_type: str, concept: str, start: float, duration: float) -> str:
    return (
        f"<ScoredEvent><EventType>{event_type}</EventType><EventConcept>{concept}</EventConcept>"
        f"<Start>{start}</Start><Duration>{duration}</Duration></ScoredEvent>"
    )


def document(*events: str) -> bytes:
    body = "".join(events)
    return (
        "<?xml version='1.0' encoding='UTF-8'?><PSGAnnotation><EpochLength>30</EpochLength>"
        f"<ScoredEvents>{body}</ScoredEvents></PSGAnnotation>"
    ).encode()


SAMPLE = document(
    "<ScoredEvent><EventType/><EventConcept>Recording Start Time</EventConcept>"
    "<Start>0</Start><Duration>36000</Duration></ScoredEvent>",
    scored("Stages|Stages", "Wake|0", 0, 1800),
    scored("Stages|Stages", "Stage 2 sleep|2", 1800, 3600),
    scored("Stages|Stages", "Stage 4 sleep|4", 5400, 1800),
    scored("Stages|Stages", "REM sleep|5", 7200, 1800),
    scored("Stages|Stages", "Unscored|9", 9000, 30),
    scored("Respiratory|Respiratory", "Obstructive apnea|Obstructive Apnea", 2000.5, 15.0),
    scored("Respiratory|Respiratory", "Hypopnea|Hypopnea", 2100, 20),
    scored("Respiratory|Respiratory", "Hypopnea|Hypopnea", 2110, 20),
    scored("Respiratory|Respiratory", "SpO2 desaturation|SpO2 desaturation", 2120, 10),
    scored("Arousals|Arousals", "Arousal|Arousal ()", 2130, 5),
)


def test_parse_stages_and_respiratory_events():
    parsed = parse_nsrr_xml(SAMPLE)
    assert [s[2] for s in parsed["stages"]] == ["W", "N2", "N3", "R"]
    assert parsed["unscored_stage_events"] == 1
    assert parsed["events"].shape == (3, 3)
    assert parsed["events"][0].tolist() == [2000.5, 2015.5, 0.0]
    assert parsed["other_concepts"]["spo2 desaturation"] == 1


def test_reference_ahi_counts_overlapping_events_separately():
    result = reference_ahi_nsrr(parse_nsrr_xml(SAMPLE))
    assert result["sleep_h"] == pytest.approx(2.0)
    assert result["event_counts"]["resp_hypopnea"] == 2
    assert result["ahi"] == pytest.approx(1.5)


def test_unknown_stage_blocks_reference():
    parsed = parse_nsrr_xml(document(scored("Stages|Stages", "Stage 9 sleep|9", 0, 30)))
    assert parsed["unknown_concepts"] == {"stage 9 sleep": 1}
    with pytest.raises(ValueError):
        reference_ahi_nsrr(parsed)


def test_epoch_stages_grid():
    labels = epoch_stages(parse_nsrr_xml(SAMPLE), 302)
    assert labels[0] == "W" and labels[60] == "N2" and labels[180] == "N3" and labels[240] == "R"
    assert labels[300] == "" and labels[301] == ""
    with pytest.raises(ValueError):
        epoch_stages(parse_nsrr_xml(document(scored("Stages|Stages", "Wake|0", 15, 30))), 4)


def test_invalid_documents_rejected():
    with pytest.raises(ValueError):
        parse_nsrr_xml(b"<Other/>")
    with pytest.raises(ValueError):
        parse_nsrr_xml(document(scored("Respiratory|Respiratory", "Hypopnea|Hypopnea", 10, 0)))
    with pytest.raises(ValueError):
        parse_nsrr_xml(document(scored("Stages|Stages", "Wake|0", 0, 60), scored("Stages|Stages", "Wake|0", 30, 30)))
    assert np.asarray(parse_nsrr_xml(document())["events"]).shape == (0, 3)


def test_unknown_respiratory_concept_blocks_reference():
    xml = document(
        scored("Stages|Stages", "Stage 2 sleep|2", 0, 7200),
        scored("Respiratory|Respiratory", "Unrecognized apnea|variant", 100, 20),
        scored("Respiratory|Respiratory", "SpO2 desaturation|SpO2 desaturation", 130, 10),
    )
    parsed = parse_nsrr_xml(xml)
    assert parsed["unknown_concepts"] == {"unrecognized apnea": 1}
    assert parsed["other_concepts"] == {"spo2 desaturation": 1}  # non ciblé, explicitement autorisé
    with pytest.raises(ValueError, match="unrecognized apnea"):
        reference_ahi_nsrr(parsed)


def test_truncated_xml_is_value_error():
    with pytest.raises(ValueError, match="XML illisible"):
        parse_nsrr_xml(document(scored("Stages|Stages", "Wake|0", 0, 30))[:-20])


def test_recording_end_from_stages():
    assert parse_nsrr_xml(SAMPLE)["recording_end_s"] == pytest.approx(9000.0)
