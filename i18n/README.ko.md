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

SONATA는 언어 콘텐츠와 감정적 소리를 모두 인식하고 전사하여 인간 표현의 교향곡을 포착하는 고급 자동 음성 인식(ASR) 시스템입니다.

## ✨ 기능

- 🎙️ WhisperX를 사용한 고정확도 음성-텍스트 변환
- 😀 523종 이상의 감정 소리와 비언어적 신호 인식
- 🌍 10개 언어 지원
- 👥 다중 화자 전사를 위한 화자 분할(온라인 및 오프라인 모드)
- ⏱️ 단어 수준의 정확한 타임스탬프 정보
- 🔄 오디오 전처리 기능

[📚 자세한 기능 문서 보기](docs/FEATURES.md)

## 🚀 설치

PyPI에서 패키지 설치:

```bash
pip install sonata-asr
```

또는 소스에서 설치:

```bash
git clone https://github.com/hwk06023/SONATA.git
cd SONATA
pip install -e .
```

## 📖 빠른 시작

### 기본 전사

```python
from sonata.core.transcriber import IntegratedTranscriber

# 전사기 초기화
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 오디오 파일 전사
result = transcriber.process_audio("path/to/audio.wav", language="en")
print(result["integrated_transcript"]["plain_text"])
```

### CLI 사용법

```bash
# 기본 사용법
sonata-asr path/to/audio.wav

# 화자 분할 기능 사용
sonata-asr path/to/audio.wav --diarize --hf-token YOUR_HUGGINGFACE_TOKEN

# 오프라인 화자 분할 사용 (설정 후 토큰 불필요)
sonata-asr path/to/audio.wav --diarize --offline-diarize --offline-config ~/.sonata/models/offline_config.yaml
```

[📚 전체 사용법 문서 보기](docs/USAGE.md)  
[⌨️ 전체 CLI 문서 보기](docs/CLI.md)  
[🎤 오프라인 다이어리제이션 가이드 보기](docs/OFFLINE_DIARIZATION.md)

## 🗣️ 지원 언어

SONATA는 영어, 한국어, 중국어, 일본어, 프랑스어, 독일어, 스페인어, 이탈리아어, 포르투갈어, 러시아어 등 10개 언어를 지원합니다.

[🌐 언어 문서 보기](docs/LANGUAGES.md)

## 🔊 오디오 이벤트 감지

SONATA는 웃음소리, 박수 소리부터 주변 소리, 음악까지 500개 이상의 다양한 오디오 이벤트를 감지할 수 있습니다.

[🎵 오디오 이벤트 문서 보기](docs/AUDIO_EVENTS.md)

## 🛣️ 로드맵

- 🌐 향상된 다국어 지원
- 🧠 고급 ASR 모델 다양화
- 😢 향상된 감정 감지
- 🔊 더 나은 화자 분할
- ⚡ 성능 최적화

[📋 전체 로드맵 보기](docs/ROADMAP.md)

## 🤝 기여하기

기여는 언제나 환영합니다! 풀 리퀘스트를 자유롭게 제출해 주세요.

[📝 기여 가이드라인 보기](docs/CONTRIBUTING.md)

## 📄 라이선스

이 프로젝트는 GNU 일반 공중 라이선스 v3.0에 따라 라이선스가 부여됩니다.

## 🙏 감사의 말

- [WhisperX](https://github.com/m-bain/whisperX) - 빠른 음성 인식
- [AudioSet AST](https://github.com/YuanGongND/ast) - 오디오 이벤트 감지
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - 화자 분할
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP 도구 