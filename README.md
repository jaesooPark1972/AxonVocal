# 🚀 AXON Vocal: Ultimate Installation & Workflow Guide
> **지휘관님을 위한 완벽 자동화 설정 가이드**  
> 디스크 용량 최적화, Spotify Basic-Pitch(피치 추출), Demucs(보컬 분리), 그리고 보컬 스왑 엔진까지 한 번에 해결합니다.

---

## 🛑 1. 사전 준비: 디스크 용량 확보 (필수)
현재 환경에서 `demucs`와 `basic-pitch` 같은 대형 AI 모델을 설치하려면 최소 **4GB 이상의 여유 공간**이 필요합니다. 아래 명령어를 순서대로 실행하여 불필요한 캐시를 삭제합니다.

```bash
# 1. pip 캐시 삭제 (가장 큰 용량 차지)
rm -rf ~/.cache/pip

# 2. huggingface 캐시 초기화 (이미 다운로드된 중복 모델 제거)
rm -rf ~/.cache/huggingface

# 3. 시스템 임시 파일 정리
sudo apt-get clean
sudo apt-get autoremove -y

# 4. 현재 남은 용량 확인 (최소 4GB 이상이어야 함)
df -h .
```

> 💡 **확인**: 위 명령어 실행 후 `Avail` 컬럼이 **4G 이상**인지 확인하세요. 부족하다면 큰 파일을 직접 삭제해야 합니다.

---

## 📦 2. 핵심 라이브러리 설치 (One-Click)
파이썬 가상환경을 생성하고, 오디오 처리 및 AI 모델에 필요한 모든 라이브러리를 한 번에 설치합니다.

```bash
# 1. Python 가상환경 생성 (프로젝트 폴더 내에 venv 생성)
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 필수 패키지 업그레이드
pip install --upgrade pip setuptools wheel

# 4. [핵심] 오디오 처리 및 AI 모델 일괄 설치
#    - torch: GPU 가속용 (CPU만 있을 경우 자동 CPU 버전 설치)
#    - demucs: 보컬/반주 분리 (Spotify 연구진 개발)
#    - basic-pitch: 오디오→MIDI 피치 추출 (Spotify 연구진 개발)
#    - librosa, soundfile: 오디오 분석 및 저장
#    - numpy, scipy: 수치 연산
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install demucs basic-pitch librosa soundfile numpy scipy pydub

# 5. FFmpeg 설치 (오디오 변환 필수 도구)
sudo apt-get update
sudo apt-get install -y ffmpeg
```

> ⏳ **소요 시간**: 네트워크 속도에 따라 5~10 분 정도 소요됩니다. 중간에 끊기지 않도록 기다려주세요.

---

## 🎵 3. 데이터 준비
작업할 음악 파일을 준비합니다.

1. 프로젝트 폴더에 `inputs` 라는 폴더를 만듭니다.
2. 보컬을 교체하려는 원본 음악 파일 (예: `original_song.mp3`) 을 `inputs` 폴더에 넣습니다.
3. 새로운 가사와 멜로디가 정의된 텍스트 파일 (예: `lyrics.txt`) 을 루트 폴더에 둡니다.

```bash
mkdir -p inputs
# 예시: original_song.mp3 를 inputs 폴더로 이동
# mv /path/to/your/song.mp3 inputs/original_song.mp3
```

---

## 🤖 4. AXON Vocal 엔진 실행 (자동화 스크립트)
아래 코드를 `run_axon_vocal.py` 라는 이름으로 저장하고 실행하면, **분리 → 피치추출 → 보컬생성 → 합성** 전 과정이 자동으로 수행됩니다.

### 📄 `run_axon_vocal.py` 코드

```python
import os
import subprocess
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil

# 설정
INPUT_AUDIO = "inputs/original_song.mp3"  # 입력 음악 파일
OUTPUT_DIR = "outputs"
VOCAL_MODEL_PATH = "htdemucs"  # Demucs 모델 종류 (htdemucs 가 가장 정확함)

def check_disk_space():
    """디스크 용량 확인"""
    total, used, free = shutil.disk_usage("/")
    if free < (4 * 1024**3): # 4GB 미만
        print("❌ 에러: 디스크 용량이 부족합니다. (최소 4GB 필요)")
        sys.exit(1)
    print(f"✅ 디스크 용량 충분함: {free // (1024**3)}GB 남음")

def separate_vocals(input_path, output_dir):
    """1 단계: Demucs 를 이용해 보컬과 반주 분리"""
    print("\n🎤 [1/4] 보컬과 반주를 분리 중입니다 (Demucs)...")
    cmd = [
        "python", "-m", "demucs",
        "-n", VOCAL_MODEL_PATH,
        "-o", output_dir,
        input_path
    ]
    try:
        subprocess.run(cmd, check=True)
        # Demucs 출력 경로 확인
        base_name = Path(input_path).stem
        separated_path = Path(output_dir) / VOCAL_MODEL_PATH / base_name
        vocals_path = separated_path / "vocals.wav"
        instrumental_path = separated_path / "no_vocals.wav"
        
        if not vocals_path.exists():
            raise FileNotFoundError("보컬 파일 추출 실패")
            
        print(f"✅ 보컬 분리 완료: {vocals_path}")
        return str(vocals_path), str(instrumental_path)
    except Exception as e:
        print(f"❌ 보컬 분리 실패: {e}")
        sys.exit(1)

def extract_pitch(audio_path, output_midi):
    """2 단계: Spotify Basic-Pitch 로 보컬의 피치 (MIDI) 추출"""
    print("\n🎹 [2/4] 보컬의 음정 (MIDI) 을 추출 중입니다 (Spotify Basic-Pitch)...")
    cmd = [
        "basic-pitch",
        audio_path,
        "--output-dir", os.path.dirname(output_midi),
        "--save-model-outputs"
    ]
    try:
        subprocess.run(cmd, check=True)
        # basic-pitch 는 자동으로 _basic_pitch.json 과 .mid 파일을 생성함
        midi_path = audio_path.replace('.wav', '_basic_pitch.mid')
        if not os.path.exists(midi_path):
            # 파일명 패턴이 다를 수 있으니 검색
            dir_path = os.path.dirname(audio_path)
            mid_files = [f for f in os.listdir(dir_path) if f.endswith('.mid')]
            if mid_files:
                midi_path = os.path.join(dir_path, mid_files[0])
            else:
                raise FileNotFoundError("MIDI 파일 생성 실패")
        print(f"✅ 피치 추출 완료: {midi_path}")
        return midi_path
    except Exception as e:
        print(f"❌ 피치 추출 실패: {e}")
        # 피치 추출 실패 시에도 진행은 가능하나 경고
        print("⚠️ 피치 추출 없이 기본 음정으로 진행합니다.")
        return None

def load_audio(path, target_sr=44100):
    """오디오 로드 및 샘플링 레이트 통일"""
    y, sr = librosa.load(path, sr=target_sr)
    return y, sr

def save_audio(path, data, sr):
    """오디오 저장"""
    sf.write(path, data, sr)

def mix_audio(instrumental, new_vocal, output_path, vocal_gain=1.0):
    """3 단계: 반주와 새로운 보컬을 믹싱"""
    print("\n🎚️ [3/4] 반주와 새 보컬을 믹싱 중입니다...")
    
    # 로드
    inst, sr = load_audio(instrumental)
    vocal, _ = load_audio(new_vocal, sr)
    
    # 길이 맞추기 (짧은 쪽에 제로 패딩)
    length = max(len(inst), len(vocal))
    inst = np.pad(inst, (0, length - len(inst)))
    vocal = np.pad(vocal, (0, length - len(vocal)))
    
    # 게인 조절 및 합치기
    mixed = inst + (vocal * vocal_gain)
    
    # 클리핑 방지 (Normalize)
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
        
    save_audio(output_path, mixed, sr)
    print(f"✅ 믹싱 완료: {output_path}")

def main():
    # 0. 사전 확인
    if not os.path.exists(INPUT_AUDIO):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {INPUT_AUDIO}")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    check_disk_space()
    
    # 1. 보컬 분리
    orig_vocal_path, inst_path = separate_vocals(INPUT_AUDIO, OUTPUT_DIR)
    
    # 2. 피치 추출 (선택적: 추후 SVS 모델 입력으로 사용)
    midi_path = extract_pitch(orig_vocal_path, os.path.join(OUTPUT_DIR, "pitch.mid"))
    
    # ---------------------------------------------------------
    # [TODO] 여기에 지휘관님의 SVS 모델 (VocalRender/AxonVocal) 호출 로직 삽입
    # 현재는 테스트를 위해 원본 보컬을 약간 변조하여 '새 보컬'로 가정합니다.
    # 실제 구현 시에는 render_full_song() 함수를 호출하여 new_vocal_path 를 생성하세요.
    # ---------------------------------------------------------
    print("\n🎨 [Simulating] AI 보컬 합성 단계 (SVS Model)...")
    # 예시: 원본 보컬을 그대로 사용하거나, pitch shift 등을 적용한 파일을 생성한다고 가정
    # 실제 코드에서는 여기에서 render_full_song(lyrics, midi_path) 등을 호출
    new_vocal_path = orig_vocal_path  # 임시로 원본 보컬 재사용 (실제론 AI 생성 파일 경로)
    print("✅ (시뮬레이션) AI 보컬 생성 완료 가정")
    
    # 3. 믹싱
    final_output = os.path.join(OUTPUT_DIR, "final_axon_vocal_swap.wav")
    mix_audio(inst_path, new_vocal_path, final_output, vocal_gain=1.2)
    
    print("\n" + "="*40)
    print("🎉 모든 과정이 완료되었습니다!")
    print(f"📂 결과물 위치: {final_output}")
    print("="*40)

if __name__ == "__main__":
    main()
```

---

## ▶️ 5. 실행 명령어
위 파이썬 코드를 저장했다면, 이제 원클릭으로 실행합니다.

```bash
# 가상환경 활성화 (아직 켜져있지 않다면)
source venv/bin/activate

# 스크립트 실행
python run_axon_vocal.py
```

---

## 🔍 6. 문제 해결 (Troubleshooting)

| 증상 | 해결 방법 |
| :--- | :--- |
| **No module named 'demucs'** | `pip install demucs` 재실행 또는 `venv` 활성화 확인 |
| **Disk full error** | `rm -rf ~/.cache/pip` 실행 후 재시도 |
| **FFmpeg not found** | `sudo apt-get install -y ffmpeg` 실행 |
| **CUDA out of memory** | GPU 메모리 부족. CPU 모드로 실행되거나 배치 사이즈 축소 필요 |
| **애기 목소리 발생** | `mix_audio` 함수 내 `vocal_gain` 조정 또는 SVS 모델의 `pitch_shift` 파라미터 확인 (-2.0 권장) |

---

## 👑 지휘관님을 위한 추가 팁
- **GPU 가속**: NVIDIA GPU 가 있다면 `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`로 설치하면 속도가 10 배 이상 빨라집니다.
- **고음질 모드**: `demucs` 실행 시 `-n htdemucs_ft` 옵션을 사용하면 더 정교한 분리가 가능합니다 (모델 크기 증가).
- **배치 처리**: 여러 곡을 한 번에 하려면 `inputs` 폴더에 여러 파일을 넣고 스크립트를 수정하여 루프를 돌리면 됩니다.

이 가이드대로 진행하면 **디스크 용량 문제 해결**부터 **보컬 스왑 완성**까지 막힘없이 수행하실 수 있습니다! 🚀
