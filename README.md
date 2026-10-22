# ovos-ww-plugin-efficientword-net

An OVOS wake-word plugin for [EfficientWord-Net](https://github.com/Ant-Brain/EfficientWord-Net), a few-shot hotword detector. You give it 4 or more recordings of a wake word, and it detects that word. It needs no training.

The plugin embeds the last 1.5 s of audio with EfficientWord-Net's ResNet-50 ArcFace model and compares the embedding with the embeddings of your recordings. A detection is a score of 0.7 or more, where the score is `(cosine similarity + 1) / 2` against the closest recording.

## Install

```bash
pip install --pre ovos-ww-plugin-efficientword-net
```

The plugin needs only `onnxruntime` and `numpy`. It carries the quantized EfficientWord-Net model (22 MB) and the upstream sample references, so it works offline and does not install PyAudio.

## Configuration

Use one of the bundled wake words: `alexa`, `balloon`, `computer`, `lights_off`, `lights_on`, `mobile` or `mycroft`.

```json
"hotwords": {
  "computer": {
    "module": "ovos-ww-plugin-efficientword-net",
    "listen": true
  }
},
"listener": {"wake_word": "computer"}
```

Or give your own recordings: a folder of 16-bit WAV files, one wake word per file, at least 4 files. Any sample rate works.

```json
"hotwords": {
  "hey_computer": {
    "module": "ovos-ww-plugin-efficientword-net",
    "samples": "~/wakewords/hey_computer",
    "listen": true
  }
}
```

| Key | Default | Meaning |
|---|---|---|
| `wakeword` | key phrase, spaces as `_` | Name of a bundled reference |
| `samples` | | Folder or list of WAV recordings; replaces the bundled reference |
| `reference_file` | | A `*_ref.json` file made by EfficientWord-Net's `generate_reference` |
| `threshold` | `0.7` | Minimum score for a detection, between 0 and 1 |
| `hop_seconds` | `0.5` | Audio between two scores |
| `relaxation_seconds` | `2.0` | Audio after a detection with no new detection |
| `model` | bundled quantized model | Path to another EfficientWord-Net ResNet-50 ArcFace ONNX file |

## Measured results

Each clip was streamed through the plugin with 1 s of silence before and after it, in chunks of 1024 samples, with the default settings and the bundled references.

| Wake word | Positive clips detected | Other clips detected |
|---|---|---|
| computer | 103 of 120 human recordings ([community dataset](https://huggingface.co/datasets/OpenVoiceOS/ovos-community-wakewords-dataset)); 93 of 120 synthetic "hey computer" | 1 of 300 speech clips ([not-wake-words-speech-en](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en)), 0 of 60 human "athena", 0 of 120 synthetic "alexa" |
| alexa | 119 of 120 synthetic ([synthetic-wakewords](https://huggingface.co/datasets/OpenVoiceOS/synthetic-wakewords)) | 0 of 300 speech clips ([not-wake-words-speech-en](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en)), 0 of 120 human "computer" |

300 one-second speech clips are not enough to estimate false activations per hour. Measure on your own audio before you use a wake word.

The quantized model gives scores within 0.005 of the full-precision model on the upstream recordings. One score takes about 50 ms on a laptop CPU, and the plugin scores every `hop_seconds`. Increase `hop_seconds` on slow hardware.

## Limits

- Detection quality depends on the recordings. Record in the room and with the microphone that the device uses.
- The bundled references are English words.
- A wake word longer than 1.5 s does not fit in the window.

## Credits

EfficientWord-Net is by ANT-BRaiN, Apache-2.0. The plugin carries its model, its sample references and a numpy version of its inference code. See [NOTICE](NOTICE) for the changes.

Report problems at [the issue tracker](https://github.com/OpenVoiceOS/ovos-ww-plugin-efficientword-net/issues).

## Funding

[![NGI0 Commons Fund](ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) programme, under the aegis of DG Communications Networks, Content and Technology under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
