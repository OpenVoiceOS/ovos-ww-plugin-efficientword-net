"""OVOS wake-word plugin for EfficientWord-Net, a few-shot hotword detector.

The engine embeds the last 1.5 s of audio with a ResNet-50 ArcFace model and
compares the embedding with reference embeddings of the wake word. The
references come from a few recordings of the wake word, so a new wake word
needs no training.

Configuration keys (under the hotword in the ``hotwords`` section of
mycroft.conf):

    wakeword (str):
        Name of a bundled reference: alexa, balloon, computer, lights_off,
        lights_on, mobile or mycroft. Defaults to the key phrase with spaces
        replaced by underscores. Ignored when ``reference_file`` or
        ``samples`` is set.

    reference_file (str):
        Path to a reference JSON file made by EfficientWord-Net
        (``python -m eff_word_net.generate_reference``).

    samples (str or list):
        A folder of WAV recordings of the wake word, or a list of WAV paths.
        At least 4 recordings. The references are computed when the plugin
        loads.

    threshold (float):
        Minimum score, between 0 and 1, for a detection. Default ``0.7``.

    hop_seconds (float):
        Audio between two scores. Default ``0.5``. Each score runs the model
        once over the last 1.5 s.

    relaxation_seconds (float):
        Audio after a detection during which no new detection is reported.
        Default ``2.0``.

    model (str):
        Path to another EfficientWord-Net ResNet-50 ArcFace ONNX file, for
        example the full-precision model. Defaults to the bundled quantized
        model.
"""
import json
import os
import wave
from os.path import dirname, expanduser, isdir, isfile, join
from typing import Any

import numpy as np
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG

from ovos_ww_plugin_efficientword.inference import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    EfficientWordModel,
    fit_window,
)

_PKG = dirname(__file__)
DEFAULT_MODEL = join(_PKG, "models", "resnet_50_arc_qint8.onnx")
REFERENCE_DIR = join(_PKG, "references")
MIN_SAMPLES = 4


def bundled_wakewords() -> list[str]:
    return sorted(f[:-len("_ref.json")] for f in os.listdir(REFERENCE_DIR) if f.endswith("_ref.json"))


def read_wav(path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV file as mono float32 at 16 kHz."""
    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = f.getframerate(), f.getnchannels()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.arange(n_out) / SAMPLE_RATE, np.arange(len(audio)) / rate, audio).astype(np.float32)
    return audio


class EfficientWordHotwordPlugin(HotWordEngine):
    """Few-shot wake-word detection with EfficientWord-Net."""

    def __init__(self, key_phrase: str = "hey mycroft", config: dict[str, Any] | None = None):
        super().__init__(key_phrase, config)
        self.threshold = float(self.config.get("threshold", 0.7))
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        self.hop_samples = max(1, int(float(self.config.get("hop_seconds", 0.5)) * SAMPLE_RATE))
        self.relaxation_samples = int(float(self.config.get("relaxation_seconds", 2.0)) * SAMPLE_RATE)

        model_path = expanduser(self.config.get("model") or DEFAULT_MODEL)
        if not isfile(model_path):
            raise FileNotFoundError(f"EfficientWord-Net model not found: {model_path}")
        self.model = EfficientWordModel(model_path)
        self.references = self._load_references()

        self._window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._since_detection = self.relaxation_samples
        self._found = False
        self.last_score = 0.0

    def _load_references(self) -> np.ndarray:
        samples = self.config.get("samples")
        if samples:
            if isinstance(samples, str):
                folder = expanduser(samples)
                if not isdir(folder):
                    raise FileNotFoundError(f"samples folder not found: {folder}")
                samples = sorted(join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav"))
            paths = [expanduser(p) for p in samples]
            if len(paths) < MIN_SAMPLES:
                raise ValueError(f"EfficientWord-Net needs at least {MIN_SAMPLES} recordings, got {len(paths)}")
            LOG.info(f"Computing EfficientWord-Net references from {len(paths)} recordings")
            return np.stack([self.model.embed(fit_window(read_wav(p))) for p in paths])

        reference_file = self.config.get("reference_file")
        if not reference_file:
            name = self.config.get("wakeword") or self.key_phrase.replace(" ", "_")
            reference_file = join(REFERENCE_DIR, f"{name}_ref.json")
            if not isfile(reference_file):
                raise ValueError(f"no bundled reference for '{name}'. Set 'samples' or 'reference_file', "
                                 f"or 'wakeword' to one of: {', '.join(bundled_wakewords())}")
        with open(expanduser(reference_file)) as f:
            data = json.load(f)
        if data.get("model_type", "resnet_50_arc") != "resnet_50_arc":
            raise ValueError(f"{reference_file}: model_type {data['model_type']} is not supported")
        references = np.asarray(data["embeddings"], dtype=np.float32)
        if references.ndim != 2 or len(references) < MIN_SAMPLES:
            raise ValueError(f"{reference_file}: expected at least {MIN_SAMPLES} embeddings")
        return references

    def update(self, chunk: bytes):
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= self.hop_samples:
            hop, self._pending = self._pending[:self.hop_samples], self._pending[self.hop_samples:]
            self._window = np.concatenate([self._window[len(hop):], hop])
            self._since_detection += len(hop)
            self.last_score = self.model.score(self.model.embed(self._window), self.references)
            if self.last_score >= self.threshold and self._since_detection >= self.relaxation_samples:
                LOG.debug(f"EfficientWord-Net detected '{self.key_phrase}' with score {self.last_score:.3f}")
                self._found = True
                self._since_detection = 0

    def found_wake_word(self) -> bool:
        found, self._found = self._found, False
        return found

    def reset(self):
        self._window[:] = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._found = False
