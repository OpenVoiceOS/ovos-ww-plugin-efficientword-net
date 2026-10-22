# Derived from EfficientWord-Net (https://github.com/Ant-Brain/EfficientWord-Net),
# Copyright 2021 ANT-BRaiN, Apache-2.0. The filterbank follows
# python_speech_features (https://github.com/jameslyons/python_speech_features), MIT.
"""EfficientWord-Net inference: log mel features, ResNet-50 ArcFace embedding, cosine score."""
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
WINDOW_SAMPLES = int(1.5 * SAMPLE_RATE)
_FRAME_LEN = 400   # 25 ms
_FRAME_STEP = 160  # 10 ms
_NFFT = 512
_NFILT = 64


def _mel_filterbank() -> np.ndarray:
    hz2mel = lambda hz: 2595 * np.log10(1 + hz / 700.0)
    mel2hz = lambda mel: 700 * (10 ** (mel / 2595.0) - 1)
    melpoints = np.linspace(hz2mel(0), hz2mel(SAMPLE_RATE / 2), _NFILT + 2)
    bins = np.floor((_NFFT + 1) * mel2hz(melpoints) / SAMPLE_RATE)
    fbank = np.zeros((_NFILT, _NFFT // 2 + 1))
    for j in range(_NFILT):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return fbank


_FBANK = _mel_filterbank()


def log_fbank(signal: np.ndarray) -> np.ndarray:
    """64-band log mel energies, 25 ms frames every 10 ms, no window, no pre-emphasis."""
    n_frames = 1 if len(signal) <= _FRAME_LEN else 1 + int(np.ceil((len(signal) - _FRAME_LEN) / _FRAME_STEP))
    padded = np.zeros((n_frames - 1) * _FRAME_STEP + _FRAME_LEN)
    padded[:len(signal)] = signal
    frames = np.lib.stride_tricks.sliding_window_view(padded, _FRAME_LEN)[::_FRAME_STEP]
    power = np.square(np.abs(np.fft.rfft(frames, _NFFT))) / _NFFT
    feat = power @ _FBANK.T
    return np.log(np.where(feat == 0, np.finfo(float).eps, feat))


def fit_window(audio: np.ndarray) -> np.ndarray:
    """Trim leading and trailing zeros, then center-pad or center-crop to 1.5 s."""
    nonzero = np.flatnonzero(audio)
    if len(nonzero):
        audio = audio[nonzero[0]:nonzero[-1] + 1]
    if len(audio) >= WINDOW_SAMPLES:
        start = (len(audio) - WINDOW_SAMPLES) // 2
        return audio[start:start + WINDOW_SAMPLES]
    out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    start = (WINDOW_SAMPLES - len(audio)) // 2
    out[start:start + len(audio)] = audio
    return out


class EfficientWordModel:
    """The ResNet-50 ArcFace embedding model of EfficientWord-Net."""

    def __init__(self, model_path: str, providers: Sequence[str] | None = None):
        self.session = ort.InferenceSession(model_path, providers=list(providers or ["CPUExecutionProvider"]))
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, window: np.ndarray) -> np.ndarray:
        """Embedding of one 1.5 s float window at 16 kHz."""
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"expected {WINDOW_SAMPLES} samples, got {window.shape}")
        features = log_fbank(window)[None, None].astype(np.float32)
        return self.session.run(None, {self.input_name: features})[0][0]

    @staticmethod
    def score(embedding: np.ndarray, references: np.ndarray) -> float:
        """Best match against the reference embeddings, as (cosine + 1) / 2."""
        return float(((references @ embedding + 1) / 2).max())
