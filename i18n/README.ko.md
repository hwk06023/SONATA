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

SONATA는 감정 표현과 비언어적 신호를 포함한 인간의 표현을 캡처하는 고급 ASR(Automatic Speech Recognition) 시스템입니다.

## ✨ Features

- 🎙️ WhisperX를 사용한 고정확도 speech-to-text 변환
- 😀 523종 이상의 emotive sound와 non-verbal cue 인식
- 🌍 99개 이상의 언어 지원
- 👥 다중 화자 전사를 위한 speaker diarization(Silero VAD와 WavLM 임베딩 활용)
- ⏱️ 단어 수준의 정확한 timestamp 정보
- 🔄 오디오 preprocessing 기능

[📚 자세한 기능 문서 보기](https://hwk06023.github.io/SONATA/FEATURES.html)

## 🚀 Installation

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

## 📖 Quick Start

### 기본 전사

```python
from sonata.core.transcriber import IntegratedTranscriber

# 전사기 초기화
transcriber = IntegratedTranscriber(asr_model="large-v3", device="cpu")

# 오디오 파일 전사
result = transcriber.process_audio("path/to/audio.wav", language="ko")
print(result["integrated_transcript"]["plain_text"])
```

### CLI 사용법

```bash
# 기본 사용법
sonata-asr path/to/audio.wav

# 화자 분할 기능 사용
sonata-asr path/to/audio.wav --diarize

# 화자 수를 알고 있는 경우 설정
sonata-asr path/to/audio.wav --diarize --num-speakers 3
```

#### 주요 CLI 옵션:

```
일반:
  -o, --output FILE           지정된 JSON 파일에 전사 결과 저장
  -l, --language LANG         언어 코드 (en, ko, zh, ja, fr, de, es, it, pt, ru)
  -m, --model NAME            WhisperX 모델 크기 (tiny, small, medium, large-v3 등)
  -d, --device DEVICE         모델 실행 장치 (cpu, cuda)
  --text-output               텍스트 파일에 전사 결과 저장 (기본값: input_name.txt)
  --preprocess                오디오 전처리 (포맷 변환 및 무음 제거)

다이어리제이션:
  --diarize                   Silero VAD와 WavLM을 사용한 SOTA 화자 다이어리제이션 활성화
  --num-speakers NUM          정확한 화자 수 설정 (선택 사항)

오디오 이벤트:
  --threshold VALUE           오디오 이벤트 감지 임계값 (0.0-1.0)
  --custom-thresholds FILE    사용자 정의 오디오 이벤트 임계값이 포함된 JSON 파일 경로
  --deep-detect               다중 스케일 오디오 이벤트 감지 활성화 (정확도 향상)
  --deep-detect-scales NUM    딥 감지를 위한 스케일 수 (1-3, 기본값: 3)
  --deep-detect-window-sizes  딥 감지를 위한 사용자 정의 윈도우 크기 (쉼표로 구분)
  --deep-detect-hop-sizes     딥 감지를 위한 사용자 정의 홉 크기 (쉼표로 구분)
```

[📚 전체 사용법 문서 보기](https://hwk06023.github.io/SONATA/USAGE.html)  
[⌨️ 전체 CLI 문서 보기](https://hwk06023.github.io/SONATA/CLI.html)  
[🎤 화자 다이어리제이션 가이드 보기](https://hwk06023.github.io/SONATA/SPEAKER_DIARIZATION.html)

## 🗣️ Supported Languages

SONATA는 Whisper large-v3를 활용하여 다양한 정확도 수준으로 99개 이상의 언어를 지원합니다. 영어, 스페인어, 프랑스어, 독일어, 일본어 등은 우수한 전사 성능(5-12% 오류율)을 보이며, 다른 언어들도 양호에서 보통 수준의 정확도를 제공합니다.

SONATA의 주요 언어 지원 특징:
- 주요 언어에 대한 뛰어난 정확도
- 중국어, 일본어, 한국어 같은 언어는 문자 기반 평가(CER) 적용
- 언어별 특성에 맞는 특화된 처리
- 다중언어 콘텐츠에 대한 고급 자동 감지 기능

[🌐 자세한 언어 지원 문서 보기](https://hwk06023.github.io/SONATA/LANGUAGES.html)

## 🔊 Audio Event Detection

SONATA는 웃음소리, 박수 소리부터 주변 소리, 음악까지 500개 이상의 다양한 오디오 이벤트를 감지할 수 있습니다. 사용자 정의 이벤트 감지 임계값 기능을 통해 팟캐스트 분석, 회의 녹취, 자연 소리 분석 등 다양한 용도에 맞게 특정 오디오 이벤트의 감도를 미세 조정할 수 있습니다.

[🎵 오디오 이벤트 문서 보기](https://hwk06023.github.io/SONATA/AUDIO_EVENTS.html)

## 🚀 Next Steps

- 🧠 고급 ASR 모델 다양화
- 😢 향상된 감정 감지
- 🔊 더 나은 speaker diarization
- ⚡ 성능 최적화
- 🛠️ 딥 감지 모드의 병렬 처리 문제 수정을 통한 안정성 향상

## 🤝 Contributing

다양한 방식의 기여를 환영합니다! SONATA는 코드 개선, 문서화, 테스트, 버그 보고 등 여러 가지 방법으로 기여할 수 있습니다. 포괄적인 기여 가이드에서 다음 내용을 다룹니다:

- 개발 환경 설정
- 코딩 표준 및 모범 사례
- 테스트 절차
- 풀 리퀘스트 워크플로우
- 문서화 가이드라인
- 언어별 고려사항

경험 많은 개발자든 오픈 소스 초보자든 모든 기여를 환영합니다.

[📝 기여 가이드라인 보기](https://hwk06023.github.io/SONATA/CONTRIBUTING.html)

## 📄 License

이 프로젝트는 GNU 일반 공중 라이선스 v3.0에 따라 라이선스가 부여됩니다.

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) - 빠른 음성 인식
- [AudioSet AST](https://github.com/YuanGongND/ast) - 오디오 이벤트 감지
  - [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) - 오디오 이벤트 분류를 위한 사전 훈련된 모델
- [Silero VAD](https://github.com/snakers4/silero-vad) - 음성 활동 감지
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - Microsoft의 고급 오디오 이해 모델
  - [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) - 화자 검증 모델
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP 도구 

## 👥 화자 분리 (Speaker Diarization)

SONATA는 녹음에서 서로 다른 화자를 식별하고 분리하는 최첨단 화자 분리 기능을 제공합니다. 이 시스템은 음성 감지에 Silero VAD를 사용하고 화자 식별에 WavLM 임베딩을 사용하여 회의, 인터뷰, 팟캐스트와 같은 다중 화자 콘텐츠 전사에 이상적입니다.

화자 분리 기능의 사용 방법:
```bash
# 기본 화자 분리
sonata-asr 오디오파일경로.wav --diarize

# 화자 수를 알고 있는 경우 지정
sonata-asr 오디오파일경로.wav --diarize --num-speakers 3

# 디버깅이나 분석을 위해 중간 단계 출력 저장
sonata-asr 오디오파일경로.wav --diarize --save-steps
```

`--save-steps` 옵션을 사용하면 SONATA는 오디오 파일 이름을 따서 만든 디렉토리에 다음과 같은 중간 파일들을 저장합니다:
- 음성 활동 감지 구간 (VAD 세그먼트)
- 화자 변경 지점
- 분석 세그먼트
- 화자 임베딩 정보
- 클러스터링 결과
- 최종 화자 세그먼트

이 기능은 특히 까다로운 오디오 파일에 대한 화자 분리 기능을 미세 조정하거나 디버깅할 때 유용합니다.

[🎙️ 화자 분리 문서 보기](https://hwk06023.github.io/SONATA/SPEAKER_DIARIZATION.html) 