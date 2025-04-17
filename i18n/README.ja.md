# SONATA 🎵🔊

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/sonata-asr.svg)](https://badge.fury.io/py/sonata-asr)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/hwk06023/SONATA?style=social)](https://github.com/hwk06023/SONATA/stargazers)

<div align="right">
<a href="../README.md">English</a> |
<a href="README.ko.md">한국어</a> |
<a href="README.zh.md">中文</a> |
<a href="README.ja.md">日本語</a>
</div>

**SOund and Narrative Advanced Transcription Assistant**

SONATAは人間の表現の交響曲を捉え、言語コンテンツと感情的な音の両方を認識して転写する高度な自動音声認識（ASR）システムです。

## ✨ 機能

- 🎙️ WhisperXを使用した高精度な音声テキスト変換
- 😀 523種類以上の感情音と非言語キューの認識
- 🌍 10言語対応
- 👥 複数話者の書き起こしのための話者分離（オンラインとオフラインモード）
- ⏱️ 単語レベルのタイムスタンプ情報
- 🔄 オーディオ前処理機能

[📚 詳細な機能ドキュメントを見る](docs/FEATURES.md)

## 🚀 インストール

PyPIからパッケージをインストール：

```bash
pip install sonata-asr
```

またはソースからインストール：

```bash
git clone https://github.com/hwk06023/SONATA.git
cd SONATA
pip install -e .
```

## 📖 クイックスタート

### 基本的な書き起こし

```python
from sonata.core.transcriber import IntegratedTranscriber

# トランスクライバーを初期化
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 音声ファイルを書き起こし
result = transcriber.process_audio("path/to/audio.wav", language="en")
print(result["integrated_transcript"]["plain_text"])
```

### CLIの使用方法

```bash
# 基本的な使用法
sonata-asr path/to/audio.wav

# 話者分離機能を使用
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN

# オフライン話者分離を使用（設定後はトークン不要）
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

[📚 完全な使用方法ドキュメントを見る](docs/USAGE.md)  
[⌨️ 完全なCLIドキュメントを見る](docs/CLI.md)  
[🎤 オフライン話者分離ガイドを見る](docs/OFFLINE_DIARIZATION.md)

## 🗣️ サポートされている言語

SONATAは英語、韓国語、中国語、日本語、フランス語、ドイツ語、スペイン語、イタリア語、ポルトガル語、ロシア語など10の言語をサポートしています。

[🌐 言語ドキュメントを見る](docs/LANGUAGES.md)

## 🔊 オーディオイベント検出

SONATAは笑い声、拍手から環境音、音楽まで500以上の異なるオーディオイベントを検出できます。

[🎵 オーディオイベントドキュメントを見る](docs/AUDIO_EVENTS.md)

## 🛣️ ロードマップ

- 🌐 多言語サポートの強化
- 🧠 高度なASRモデルの多様化
- 😢 感情検出の改善
- 🔊 より良い話者分離
- ⚡ パフォーマンスの最適化

[📋 完全なロードマップを見る](docs/ROADMAP.md)

## 🤝 貢献

貢献は大歓迎です！気軽にプルリクエストを送信してください。

[📝 貢献ガイドラインを見る](docs/CONTRIBUTING.md)

## 📄 ライセンス

このプロジェクトはGNU一般公衆ライセンスv3.0の下でライセンスされています。

## 🙏 謝辞

- [WhisperX](https://github.com/m-bain/whisperX) - 高速音声認識
- [AudioSet AST](https://github.com/YuanGongND/ast) - オーディオイベント検出
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - 話者分離
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLPツール 