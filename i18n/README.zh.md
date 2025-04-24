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

SONATA 是一个先进的 ASR(Automatic Speech Recognition) 系统，能够捕捉包括情感声音和非语言线索在内的人类表达。

## ✨ Features

- 🎙️ 使用 WhisperX 的高精度 speech-to-text 转换
- 😀 识别 523+ 种 emotive sound 和 non-verbal cue
- 🌍 支持 10 种语言
- 👥 支持多说话人转录的 speaker diarization（在线和离线模式）
- ⏱️ 单词级的精确 timestamp 信息
- 🔄 音频 preprocessing 功能

[📚 查看详细功能文档](../docs/FEATURES.md)

## 🚀 Installation

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

### 命令行使用

```bash
# 基本用法
sonata-asr path/to/audio.wav

# 使用说话人分离
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN

# 使用离线说话人分离（设置后无需令牌）
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

> **注意:** 在线说话人分离需要获得 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 和 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 两个模型的访问权限。请访问这两个模型页面并接受各自的使用条款以获取访问权限。

#### 常用 CLI 选项:

```
一般选项:
  -o, --output FILE           将转录结果保存到指定的 JSON 文件
  -l, --language LANG         语言代码 (en, ko, zh, ja, fr, de, es, it, pt, ru)
  -m, --model NAME            WhisperX 模型大小 (tiny, small, medium, large-v3 等)
  -d, --device DEVICE         运行模型的设备 (cpu, cuda)
  --text-output FILE          将格式化的转录结果保存到指定的文本文件
  --format TYPE               输出格式: concise, default, extended
  --preprocess                音频预处理 (格式转换和静音剪裁)

说话人分离:
  --diarize                   启用说话人分离
  --hf-token TOKEN            HuggingFace 令牌 (用于在线说话人分离)
  --min-speakers NUM          设置最小说话人数量
  --max-speakers NUM          设置最大说话人数量
  --offline-diarize           使用离线说话人分离 (设置后无需令牌)
  --offline-config PATH       离线说话人分离配置文件路径
  --setup-offline             下载并设置离线说话人分离模型

音频事件:
  --threshold VALUE           音频事件检测阈值 (0.0-1.0)
  --custom-thresholds FILE    自定义音频事件阈值的 JSON 文件路径
```

[📚 查看完整使用文档](../docs/USAGE.md)  
[⌨️ 查看完整命令行文档](../docs/CLI.md)  
[🎤 查看离线说话人分离指南](../docs/OFFLINE_DIARIZATION.md)

## 🗣️ Supported Languages

SONATA 支持 10 种语言，包括英语、韩语、中文、日语、法语、德语、西班牙语、意大利语、葡萄牙语和俄语。

[🌐 查看语言文档](../docs/LANGUAGES.md)

## 🔊 Audio Event Detection

SONATA 可以检测 500 多种不同的音频事件，从笑声、掌声到环境声音和音乐。通过自定义事件检测阈值功能，您可以针对特定音频事件调整敏感度，以适应播客分析、会议记录或自然声音分析等各种用途。

[🎵 查看音频事件文档](../docs/AUDIO_EVENTS.md)

## 🚀 Next Steps

- 🧠 丰富高级 ASR 模型多样性
- 😢 提升情感检测能力
- 🔊 改进 speaker diarization 效果
- ⚡ 优化性能表现

## 🤝 Contributing

Contributing 欢迎！请随时提交拉取请求。

[📝 查看贡献指南](../docs/CONTRIBUTING.md)

## 📄 License

本项目采用 GNU 通用公共许可证 v3.0 授权。

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - 快速语音识别
- [AudioSet AST](https://github.com/YuanGongND/ast) - 音频事件检测
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - 用于音频事件分类的预训练模型
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - speaker diarization
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) - 说话人分离流水线
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP 工具 