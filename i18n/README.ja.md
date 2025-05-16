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

SONATAは、感情表現や非言語的信号を含む人間の表現をキャプチャする高度なASR（Automatic Speech Recognition）システムです。

## ✨ Features

- 🎙️ WhisperXを使用した高精度の speech-to-text 変換
- 😀 523種類以上の emotive sound と non-verbal cue の認識
- 🌍 99以上の言語をサポート
- 👥 複数の話者の書き起こしのための speaker diarization（オンラインおよびオフラインモード）
- ⏱️ 単語レベルの正確な timestamp 情報
- 🔄 オーディオ preprocessing 機能

[📚 詳細な機能のドキュメントを見る](https://hwk06023.github.io/SONATA/FEATURES.html)

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

# トランスクライバーの初期化
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# オーディオファイルの書き起こし
result = transcriber.process_audio("path/to/audio.wav", language="ja")
print(result["integrated_transcript"]["plain_text"])
```

### CLI の使用方法

```bash
# 基本的な使用法
sonata-asr path/to/audio.wav

# speaker diarization 機能を使用
sonata-asr path/to/audio.wav --diarize

# オーディオイベント検出機能を使用
sonata-asr path/to/audio.wav --audio

# 話者数がわかっている場合は設定
sonata-asr path/to/audio.wav --diarize --num-speakers 3

# 話者分離とオーディオイベント検出を組み合わせて使用
sonata-asr path/to/audio.wav --diarize --audio
```

#### 主な CLI オプション：

```
一般：
  -o, --output FILE           指定されたJSONファイルに書き起こし結果を保存
  -l, --language LANG         言語コード（en, ko, zh, ja, fr, de, es, it, pt, ru）
  -m, --model NAME            WhisperXモデルサイズ（tiny, small, medium, large-v3など）
  -d, --device DEVICE         モデル実行デバイス（cpu, cuda）
  --text-output               テキストファイルに書き起こし結果を保存（デフォルト：input_name.txt）
  --preprocess                オーディオ preprocessing（フォーマット変換と無音除去）

Diarization：
  --diarize                   Silero VADとWavLMを使用したSOTA speaker diarization を有効化
  --num-speakers NUM          正確な話者数を設定（オプション）

Audio Event：
  --audio                     オーディオイベント検出を有効化
  --threshold VALUE           オーディオイベント検出のしきい値（0.0-1.0）
  --custom-thresholds FILE    カスタムオーディオイベントしきい値を含むJSONファイルのパス
  --deep-detect               マルチスケールオーディオイベント検出を有効化（精度向上）
  --deep-detect-scales NUM    deep detection のためのスケール数（1-3、デフォルト：3）
  --deep-detect-window-sizes  deep detection のためのカスタムウィンドウサイズ（カンマ区切り）
  --deep-detect-hop-sizes     deep detection のためのカスタムホップサイズ（カンマ区切り）
```

[📚 全使用法ドキュメントを見る](https://hwk06023.github.io/SONATA/USAGE.html)  
[⌨️ 全CLIドキュメントを見る](https://hwk06023.github.io/SONATA/CLI.html)  
[🎤 話者分離ガイドを見る](https://hwk06023.github.io/SONATA/SPEAKER_DIARIZATION.html)

## 🗣️ Supported Languages

SONATAはWhisper large-v3を活用して、様々な精度レベルで99以上の言語をサポートしています。英語、スペイン語、フランス語、ドイツ語、日本語などは優れた書き起こし性能（5-12%のエラー率）を示し、その他の言語も良好から普通レベルの精度を提供します。

SONATAの主な言語サポート特徴：
- 主要言語に対する優れた精度
- 中国語、日本語、韓国語などの言語には文字ベースの評価（CER）を適用
- 言語別特性に合わせた特化した処理
- 多言語コンテンツに対する高度な自動検出機能

[🌐 詳細な言語サポートドキュメントを見る](https://hwk06023.github.io/SONATA/LANGUAGES.html)

## 🔊 Audio Event Detection

SONATAは、笑い声、拍手音から環境音、音楽まで500以上の様々なオーディオイベントを検出できます。カスタムイベント検出しきい値機能を通じて、ポッドキャスト分析、会議録音、自然音分析など様々な用途に合わせて特定のオーディオイベントの感度を微調整できます。

[🎵 オーディオイベントドキュメントを見る](https://hwk06023.github.io/SONATA/AUDIO_EVENTS.html)

## 🚀 Next Steps

- 🧠 高度な ASR モデルの多様化
- 😢 向上した emotion detection
- 🔊 より良い speaker diarization
- ⚡ パフォーマンス最適化
- 🛠️ deep detection モードの並列処理問題修正による安定性向上

## 🤝 Contributing

様々な方法での貢献を歓迎します！SONATAはコード改善、文書化、テスト、バグ報告など様々な方法で貢献できます。包括的な貢献ガイドでは以下の内容を扱います：

- 開発環境のセットアップ
- コーディング標準とベストプラクティス
- テスト手順
- プルリクエストワークフロー
- 文書化ガイドライン
- 言語別の考慮事項

経験豊富な開発者もオープンソース初心者も、すべての貢献を歓迎します。

[📝 貢献ガイドラインを見る](https://hwk06023.github.io/SONATA/CONTRIBUTING.html)

## 📄 License

このプロジェクトはGNU一般公衆ライセンスv3.0の下でライセンスされています。

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - 高速音声認識
- [AudioSet AST](https://github.com/YuanGongND/ast) - オーディオイベント検出
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - オーディオイベント分類のための事前訓練モデル
- [Silero VAD](https://github.com/snakers4/silero-vad) - 音声活動検出
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - Microsoftの高度オーディオ理解モデル
  - [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) - スピーカー検証モデル
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLPツール 

## 👥 話者分離（Speaker Diarization）

SONATAは、録音内の異なる話者を識別し区別するための最先端の話者分離機能を提供します。このシステムは音声検出にSilero VADを使用し、話者識別にWavLM埋め込みを使用しており、会議、インタビュー、ポッドキャストなどの複数話者コンテンツの書き起こしに理想的です。

話者分離機能の使い方は簡単です：
```bash
# 基本的な分離
sonata-asr オーディオファイルのパス.wav --diarize

# 話者数がわかっている場合は指定
sonata-asr オーディオファイルのパス.wav --diarize --num-speakers 3

# デバッグや分析のために中間ステップ出力を保存
sonata-asr オーディオファイルのパス.wav --diarize --save-steps
```

`--save-steps`オプションを使用すると、SONATAはオーディオファイル名に基づいたディレクトリに以下の中間ファイルを保存します：
- 音声活動検出セグメント（VADセグメント）
- 話者変更ポイント
- 分析セグメント
- 話者埋め込み情報
- クラスタリング結果
- 最終話者セグメント

これは、特に複雑なオーディオファイルの話者分離機能を微調整またはデバッグする際に非常に役立ちます。

[🎙️ 話者分離のドキュメントを見る](https://hwk06023.github.io/SONATA/SPEAKER_DIARIZATION.html) 