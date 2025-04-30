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

**声音和叙事高级转录助手 (SOund and Narrative Advanced Transcription Assistant)**

SONATA是一个先进的ASR（自动语音识别）系统，能够捕捉人类表达中的情感表现和非语言信号。

## ✨ Features

- 🎙️ 使用WhisperX进行高精度 speech-to-text 转换
- 😀 识别523+种 emotive sound 和 non-verbal cue
- 🌍 支持99+种语言
- 👥 多说话人转录的 speaker diarization（在线和离线模式）
- ⏱️ 精确到单词级别的 timestamp 信息
- 🔄 音频 preprocessing 功能

[📚 查看详细功能文档](https://hwk06023.github.io/SONATA/FEATURES.html)

## 🚀 Installation

通过PyPI安装软件包：

```bash
pip install sonata-asr
```

或从源代码安装：

```bash
git clone https://github.com/hwk06023/SONATA.git
cd SONATA
pip install -e .
```

## 📖 Quick Start

### 基本转录

```python
from sonata.core.transcriber import IntegratedTranscriber

# 初始化转录器
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 转录音频文件
result = transcriber.process_audio("path/to/audio.wav", language="zh")
print(result["integrated_transcript"]["plain_text"])
```

### 使用 CLI

```bash
# 基本用法
sonata-asr path/to/audio.wav

# 使用 speaker diarization 功能
sonata-asr path/to/audio.wav --diarize

# 当知道说话人数量时设置
sonata-asr path/to/audio.wav --diarize --num-speakers 3
```

#### 主要 CLI 选项：

```
通用：
  -o, --output FILE           将转录结果保存到指定的JSON文件
  -l, --language LANG         语言代码（en, ko, zh, ja, fr, de, es, it, pt, ru）
  -m, --model NAME            WhisperX模型大小（tiny, small, medium, large-v3等）
  -d, --device DEVICE         模型运行设备（cpu, cuda）
  --text-output               将转录结果保存到文本文件（默认：input_name.txt）
  --preprocess                音频 preprocessing（格式转换和静音移除）

Diarization：
  --diarize                   启用使用Silero VAD和WavLM的SOTA speaker diarization
  --num-speakers NUM          设置准确的说话人数量（可选）

Audio Event：
  --threshold VALUE           音频事件检测阈值（0.0-1.0）
  --custom-thresholds FILE    包含自定义音频事件阈值的JSON文件路径
  --deep-detect               启用多尺度音频事件检测（提高准确性）
  --deep-detect-scales NUM    deep detection 的尺度数量（1-3，默认：3）
  --deep-detect-window-sizes  deep detection 的自定义窗口大小（逗号分隔）
  --deep-detect-hop-sizes     deep detection 的自定义跳跃大小（逗号分隔）
```

[📚 查看完整使用文档](https://hwk06023.github.io/SONATA/USAGE.html)  
[⌨️ 查看完整CLI文档](https://hwk06023.github.io/SONATA/CLI.html)  
[🎤 查看说话人分离指南](https://hwk06023.github.io/SONATA/SPEAKER_DIARIZATION.html)

## 🗣️ Supported Languages

SONATA利用Whisper large-v3支持99+种不同精度水平的语言。英语、西班牙语、法语、德语和日语等展示出优秀的转录性能（5-12%的错误率），其他语言也提供从良好到中等的精度。

SONATA的主要语言支持特点：
- 主要语言的出色准确度
- 对中文、日语、韩语等语言应用基于字符的评估（CER）
- 针对语言特定特性的专门处理
- 对多语言内容的高级自动检测

[🌐 查看详细语言支持文档](https://hwk06023.github.io/SONATA/LANGUAGES.html)

## 🔊 Audio Event Detection

SONATA能够检测500多种不同的音频事件，从笑声、掌声到环境声音和音乐。通过自定义事件检测阈值功能，可以针对不同应用场景（如播客分析、会议录音、自然声音分析）调整特定音频事件的敏感度。

[🎵 查看音频事件文档](https://hwk06023.github.io/SONATA/AUDIO_EVENTS.html)

## 🚀 Next Steps

- 🧠 多样化高级 ASR 模型
- 😢 增强 emotion detection
- 🔊 改进 speaker diarization
- ⚡ 性能优化
- 🛠️ 通过修复 deep detection 模式的并行处理问题提高稳定性

## 🤝 Contributing

我们欢迎多种形式的贡献！您可以通过多种方式为SONATA做出贡献，包括代码改进、文档编写、测试和错误报告。我们的综合贡献指南涵盖：

- 开发环境设置
- 编码标准和最佳实践
- 测试程序
- 拉取请求工作流程
- 文档指南
- 语言特定考虑因素

无论您是经验丰富的开发者还是开源新手，我们都欢迎所有贡献。

[📝 查看贡献指南](https://hwk06023.github.io/SONATA/CONTRIBUTING.html)

## 📄 License

本项目采用GNU通用公共许可证v3.0授权。

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - 快速语音识别
- [AudioSet AST](https://github.com/YuanGongND/ast) - 音频事件检测
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - 用于音频事件分类的预训练模型
- [Silero VAD](https://github.com/snakers4/silero-vad) - 语音活动检测
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - 微软高级音频理解模型
  - [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) - 说话人验证模型
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP工具 