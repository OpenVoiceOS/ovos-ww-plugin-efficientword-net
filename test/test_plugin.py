import json

import numpy as np
import pytest

from ovos_ww_plugin_efficientword import EfficientWordHotwordPlugin, bundled_wakewords


def test_bundled_wakewords():
    assert bundled_wakewords() == ["alexa", "balloon", "computer", "lights_off", "lights_on", "mobile", "mycroft"]


def test_key_phrase_selects_bundled_reference():
    engine = EfficientWordHotwordPlugin("lights on", {})
    assert engine.references.shape == (7, 2048)


def test_unknown_wakeword_lists_the_bundled_ones():
    with pytest.raises(ValueError, match="alexa, balloon, computer"):
        EfficientWordHotwordPlugin("hey jarvis", {})


@pytest.mark.parametrize("threshold", [0, 1, 1.5, -0.2])
def test_threshold_out_of_range(threshold):
    with pytest.raises(ValueError):
        EfficientWordHotwordPlugin("alexa", {"threshold": threshold})


def test_too_few_samples(tmp_path):
    with pytest.raises(ValueError, match="at least 4"):
        EfficientWordHotwordPlugin("x", {"samples": [str(tmp_path / f"{i}.wav") for i in range(3)]})


def test_reference_file_with_other_model_type(tmp_path):
    ref = tmp_path / "x_ref.json"
    ref.write_text(json.dumps({"embeddings": [[0.0] * 2048] * 4, "model_type": "first_iteration_siamese"}))
    with pytest.raises(ValueError, match="not supported"):
        EfficientWordHotwordPlugin("x", {"reference_file": str(ref)})


def test_silence_does_not_trigger_and_latch_is_read_once():
    engine = EfficientWordHotwordPlugin("alexa", {"hop_seconds": 0.5})
    engine.update(np.zeros(16000 * 3, dtype=np.int16).tobytes())
    assert engine.found_wake_word() is False
    engine._found = True
    assert engine.found_wake_word() is True
    assert engine.found_wake_word() is False


def test_scores_once_per_hop_whatever_the_chunk_size():
    engine = EfficientWordHotwordPlugin("alexa", {"hop_seconds": 0.25})
    calls = []
    embed = engine.model.embed
    engine.model.embed = lambda w: calls.append(1) or embed(w)
    pcm = (np.random.default_rng(0).standard_normal(16000) * 1000).astype(np.int16).tobytes()
    for i in range(0, len(pcm), 300):
        engine.update(pcm[i:i + 300])
    assert len(calls) == 4


def test_reset_clears_buffered_audio():
    engine = EfficientWordHotwordPlugin("alexa", {})
    engine.update((np.ones(30000, dtype=np.int16) * 1000).tobytes())
    engine._found = True
    engine.reset()
    assert not engine._window.any() and len(engine._pending) == 0
    assert engine.found_wake_word() is False
