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

SONATA 是一个先进的自动语音识别(ASR)系统，它能捕捉人类表达的交响乐，识别并转录语言内容和情感声音。

## ✨ 功能

- 🎙️ 使用 WhisperX 的高准确度语音转文本
- 😀 识别 523+ 种情感声音和非语言线索
- 🌍 支持 10 种语言
- 👥 支持多发言者转录的说话人分割（在线和离线模式）
- ⏱️ 词级时间戳信息
- 🔄 音频预处理功能

[📚 查看详细功能文档](docs/FEATURES.md)

## 🚀 安装

从 PyPI 安装：

```bash
pip install sonata-asr
```

或从源代码安装：

```bash
git clone https://github.com/hwk06023/SONATA.git
cd SONATA
pip install -e .
```

## 📖 快速开始

### 基本转录

```python
from sonata.core.transcriber import IntegratedTranscriber

# 初始化转录器
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 转录音频文件
result = transcriber.process_audio("path/to/audio.wav", language="en")
print(result["integrated_transcript"]["plain_text"])
```

### 命令行使用

```bash
# 基本用法
sonata-asr path/to/audio.wav

# 使用说话人分割
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN

# 使用离线说话人分割（设置后无需令牌）
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

[📚 查看完整使用文档](docs/USAGE.md)  
[⌨️ 查看完整命令行文档](docs/CLI.md)  
[🎤 查看离线说话人分割指南](docs/OFFLINE_DIARIZATION.md)

## 🗣️ 支持的语言

SONATA 支持 10 种语言，包括英语、韩语、中文、日语、法语、德语、西班牙语、意大利语、葡萄牙语和俄语。

[🌐 查看语言文档](docs/LANGUAGES.md)

## 🔊 音频事件检测

SONATA 可以检测 500 多种不同的音频事件，从笑声、掌声到环境声音、音乐等。

[🎵 查看音频事件文档](docs/AUDIO_EVENTS.md)

## 🛣️ 路线图

- 🌐 增强多语言支持
- 🧠 高级 ASR 模型多样性
- 😢 改进情感检测
- 🔊 更好的说话人分割
- ⚡ 性能优化

[📋 查看完整路线图](docs/ROADMAP.md)

## 🤝 贡献

欢迎贡献！请随时提交拉取请求。

[📝 查看贡献指南](docs/CONTRIBUTING.md)

## 📄 许可证

本项目采用 GNU 通用公共许可证 v3.0 授权。

## 🙏 致谢

- [WhisperX](https://github.com/m-bain/whisperX) - 快速语音识别
- [AudioSet AST](https://github.com/YuanGongND/ast) - 音频事件检测
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - 说话人分割
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP 工具 