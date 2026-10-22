"""Wake-word detection through the OPM factory and ovoscope's MiniListener, on real audio.

Clips (see test/clips/README.md): alexa_human.wav is a human recording from
EfficientWord-Net; alexa_synthetic/ comes from OpenVoiceOS/synthetic-wakewords
and negative/ from TigreGotico/not-wake-words-speech-en (both CC-BY-4.0).
"""
from pathlib import Path

import numpy as np
from ovos_plugin_manager.wakewords import OVOSWakeWordFactory, find_wake_word_plugins
from ovoscope.listener import get_mini_listener

from ovos_ww_plugin_efficientword import read_wav

CLIPS = Path(__file__).parent / "clips"
PLUGIN = "ovos-ww-plugin-efficientword-net"
SYNTHETIC = sorted((CLIPS / "alexa_synthetic").glob("*.wav"))
NEGATIVES = sorted((CLIPS / "negative").glob("*.wav"))


def pcm(*paths, gap=0.5):
    silence = np.zeros(int(gap * 16000), dtype=np.float32)
    parts = [silence]
    for p in paths:
        parts += [read_wav(str(p)), silence]
    return (np.concatenate(parts) * 32767).astype(np.int16).tobytes()


def scan(config, audio):
    engine = OVOSWakeWordFactory.create_hotword("alexa", {"hotwords": {"alexa": {"module": PLUGIN, **config}}})
    assert type(engine).__module__.startswith("ovos_ww_plugin_efficientword")
    listener = get_mini_listener(ww_instances={"alexa": engine})
    try:
        return listener.scan_for_wakeword(audio, frame_size=2048)
    finally:
        listener.shutdown()


def test_entry_point_is_discovered():
    assert PLUGIN in find_wake_word_plugins()


def test_bundled_reference_detects_a_human_recording():
    detected, frame = scan({}, pcm(CLIPS / "alexa_human.wav"))
    assert detected
    # the word starts after 0.5 s of silence: no detection can come from the silence before it
    assert frame * 1024 / 16000 > 0.5


def test_bundled_reference_ignores_other_speech():
    assert scan({}, pcm(*NEGATIVES)) == (False, None)


def test_references_from_recordings_detect_a_new_voice():
    detected, _ = scan({"samples": [str(p) for p in SYNTHETIC[:4]]}, pcm(SYNTHETIC[4]))
    assert detected


def test_references_from_recordings_ignore_other_speech():
    assert scan({"samples": [str(p) for p in SYNTHETIC[:4]]}, pcm(*NEGATIVES)) == (False, None)


def test_relaxation_reports_one_detection_per_utterance():
    engine = OVOSWakeWordFactory.create_hotword("alexa", {"hotwords": {"alexa": {"module": PLUGIN}}})
    audio = pcm(CLIPS / "alexa_human.wav")
    hits = 0
    for i in range(0, len(audio), 2048):
        engine.update(audio[i:i + 2048])
        hits += engine.found_wake_word()
    assert hits == 1
