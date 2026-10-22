import numpy as np
import pytest
from python_speech_features import logfbank

from ovos_ww_plugin_efficientword.inference import WINDOW_SAMPLES, fit_window, log_fbank


@pytest.mark.parametrize("n", [WINDOW_SAMPLES, 16000, 400, 401, 30001])
def test_log_fbank_matches_python_speech_features(n):
    signal = np.random.default_rng(n).standard_normal(n) * 0.1
    expected = logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=64, nfft=512, preemph=0.0)
    assert log_fbank(signal).shape == expected.shape
    np.testing.assert_allclose(log_fbank(signal), expected, rtol=1e-10, atol=1e-10)


def test_fit_window_centers_short_audio():
    audio = np.ones(1000, dtype=np.float32)
    window = fit_window(np.concatenate([np.zeros(50), audio, np.zeros(70)]))
    assert window.shape == (WINDOW_SAMPLES,)
    start = (WINDOW_SAMPLES - 1000) // 2
    assert window[start:start + 1000].sum() == 1000
    assert window.sum() == 1000


def test_fit_window_center_crops_long_audio():
    audio = np.arange(1, WINDOW_SAMPLES + 101, dtype=np.float32)
    window = fit_window(audio)
    assert window[0] == 51 and window[-1] == WINDOW_SAMPLES + 50
