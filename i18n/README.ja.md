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

SONATAは、感情表現や非言語的キューを含む人間の表現をキャプチャする先進的なASR(Automatic Speech Recognition)システムです。

## ✨ Features

- 🎙️ WhisperXを使用した高精度 speech-to-text 変換
- 😀 523種類以上の emotive sound と non-verbal cue の認識
- 🌍 10言語対応
- 👥 複数話者の書き起こしのための speaker diarization（オンラインとオフラインモード）
- ⏱️ 単語レベルの正確な timestamp 情報
- 🔄 オーディオ preprocessing 機能

[📚 詳細な機能ドキュメントを見る](../docs/FEATURES.md)

## 🚀 Installation

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

## 📖 Quick Start

### 基本的な書き起こし

```python
from sonata.core.transcriber import IntegratedTranscriber

# トランスクライバーを初期化
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 音声ファイルを書き起こし
result = transcriber.process_audio("path/to/audio.wav", language="ja")
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

> **注意:** オンライン話者分離を使用する場合、[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) と [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) の両方のモデルのアクセス権限が必要です。両方のモデルページを訪問し、それぞれの利用規約に同意してアクセス権を取得してください。

#### 主なCLIオプション:

```
一般:
  -o, --output FILE           指定したJSONファイルに書き起こし結果を保存
  -l, --language LANG         言語コード (en, ko, zh, ja, fr, de, es, it, pt, ru)
  -m, --model NAME            WhisperXモデルサイズ (tiny, small, medium, large-v3 など)
  -d, --device DEVICE         モデル実行デバイス (cpu, cuda)
  --text-output FILE          整形された書き起こしをテキストファイルに保存
  --format TYPE               出力形式: concise, default, extended
  --preprocess                オーディオの前処理（形式変換と無音トリミング）

話者分離:
  --diarize                   話者分離を有効化
  --hf-token TOKEN            HuggingFaceトークン（オンライン話者分離用）
  --min-speakers NUM          最小話者数を設定
  --max-speakers NUM          最大話者数を設定
  --offline-diarize           オフライン話者分離を使用（設定後はトークン不要）
  --offline-config PATH       オフライン話者分離設定ファイルのパス
  --setup-offline             オフライン話者分離モデルのダウンロードとセットアップ

音声イベント:
  --threshold VALUE           音声イベント検出の閾値 (0.0-1.0)
  --custom-thresholds FILE    カスタム音声イベント閾値を含むJSONファイルのパス
```

[📚 完全な使用方法ドキュメントを見る](../docs/USAGE.md)  
[⌨️ 完全なCLIドキュメントを見る](../docs/CLI.md)  
[🎤 オフライン diarization ガイドを見る](../docs/OFFLINE_DIARIZATION.md)

## 🗣️ Supported Languages

SONATAは英語、韓国語、中国語、日本語、フランス語、ドイツ語、スペイン語、イタリア語、ポルトガル語、ロシア語など10の言語をサポートしています。

[🌐 言語ドキュメントを見る](../docs/LANGUAGES.md)

## 🔊 Audio Event Detection

SONATAは笑い声、拍手から環境音、音楽まで500以上の異なるオーディオイベントを検出できます。カスタマイズ可能なイベント検出閾値機能により、ポッドキャスト分析、会議録音、自然音分析など様々な用途に合わせて特定のオーディオイベントの感度を微調整することができます。

[🎵 オーディオイベントドキュメントを見る](../docs/AUDIO_EVENTS.md)

## 🚀 Next Steps

- 🧠 高度なASRモデルの多様化
- 😢 感情検出の改善
- 🔊 より優れた speaker diarization
- ⚡ パフォーマンスの最適化

## 🤝 Contributing

Contributing 大歓迎です！気軽にプルリクエストを送信してください。

[📝 貢献ガイドラインを見る](../docs/CONTRIBUTING.md)

## 📄 License

このプロジェクトはGNU一般公衆ライセンスv3.0の下でライセンスされています。

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - 高速音声認識
- [AudioSet AST](https://github.com/YuanGongND/ast) - オーディオイベント検出
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - オーディオイベント分類のための事前学習モデル
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - speaker diarization
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) - 話者ダイアリゼーションパイプライン
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLPツール 