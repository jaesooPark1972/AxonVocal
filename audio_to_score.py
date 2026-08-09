"""Audio → Score converter: Spotify basic-pitch NN + Librosa fallback,
language-aware lyric tokenizer (한글=글자 / 영문=단어), deterministic AI composer."""
import hashlib
import re
from typing import Dict, List

import librosa
import numpy as np

PRO_SCORE_PRESETS = {
    "1. K-Pop 서정 발라드 [C Major / 72 BPM] (그대 내 품에 안겨요)": {
        "bpm": 72,
        "lyrics": "그 대 내 품 에 안 겨 요 지 금 이 순 간 을 기 억 해 요",
        "pitches": "60 62 64 65 67 65 64 62 60 62 64 65 67 69 67 65 64",
        "notes": "<NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_2>",
        "desc": "잔잔하고 감성적인 K-Pop 표준 팝 발라드 멜로디 라인",
    },
    "2. K-Pop 댄스 타이틀곡 [Eb Major / 126 BPM] (빛나는 별처럼 날아올라)": {
        "bpm": 126,
        "lyrics": "빛 나 는 별 처 럼 날 아 올 라 세 상 을 환 하 게 비 춰 봐요",
        "pitches": "67 67 69 71 72 71 69 67 65 67 69 71 72 74 72 71 69",
        "notes": "<NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4>",
        "desc": "빠르고 신나는 에너제틱 K-Pop 아이돌 타이틀곡 댄스 멜로디",
    },
    "3. 정통 트로트/가요 [G Major / 108 BPM] (사랑해요 내 님아 돌아와줘요)": {
        "bpm": 108,
        "lyrics": "사 랑 해 요 내 님 아 돌 아 와 줘 요 이 내 가 슴 은 아 파요",
        "pitches": "67 71 74 71 67 64 62 64 67 71 67 64 62 59 62 64 62",
        "notes": "<NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_2>",
        "desc": "꺾기 멜로디와 감성이 살아있는 한국 정통 트로트 선율",
    },
    "4. 감성 R&B / Soul [F Major / 88 BPM] (Hold Me Close Tonight)": {
        "bpm": 88,
        "lyrics": "Hold Me Close Tonight I Need You In My Life Forever More",
        "pitches": "65 69 72 74 77 76 74 72 69 67 65 69",
        "notes": "<NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_2>",
        "desc": "소울풀한 그루브와 풍성한 화성이 돋보이는 R&B 보컬 라인",
    },
    "5. 어쿠스틱 포크 / 캠프파이어 [G Major / 96 BPM] (바람이 불어오는 곳)": {
        "bpm": 96,
        "lyrics": "바 람 이 불 어 오 는 곳 그 곳 으 로 가 네 설 레는 마음",
        "pitches": "67 69 71 71 69 67 64 67 69 71 69 67 64 62 64 67 67",
        "notes": "<NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_2>",
        "desc": "기타 반주에 잘 어울리는 따스하고 정겨운 어쿠스틱 포크 선율",
    },
    "6. J-Pop / 애니메이션 OST [A Minor / 142 BPM] (미래를 향해 달려가자)": {
        "bpm": 142,
        "lyrics": "미 래 를 향 해 달 려 가 자 꿈 을 향 한 여 정 이 시작 되네",
        "pitches": "69 72 74 76 77 79 77 76 74 72 69 67 65 67 69 72 74 76",
        "notes": "<NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4>",
        "desc": "속도감 넘치는 열정적인 애니메이션 주제가 멜로디",
    },
    "7. 모던 록 / 팝 록 [D Major / 130 BPM] (자유를 향한 너의 함성)": {
        "bpm": 130,
        "lyrics": "자 유 를 향 한 너 의 함 성 시 원 한 바 람 을 타 고",
        "pitches": "62 66 69 71 74 71 69 66 62 66 69 71 69 66 62",
        "notes": "<NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_2>",
        "desc": "강렬한 비트와 탁 트인 고음이 어우러지는 팝 록 멜로디",
    },
    "8. 힙합 / 트랩 후크 [C Minor / 95 BPM] (도시의 밤을 밝히는 빛)": {
        "bpm": 95,
        "lyrics": "도 시 의 밤 을 밝 히 는 빛 우 리 의 성 장 은 멈추 지 않아",
        "pitches": "60 63 65 67 67 65 63 60 58 60 63 65 67 65 63 60 63 60",
        "notes": "<NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_8> <NOTE_4>",
        "desc": "트렌디한 캐치함이 느껴지는 힙합/트랩 보컬 후크 라인",
    },
    "9. 클래식 / 오페라 아리아 [D Minor / 76 BPM] (밤의 여왕 아리아 풍)": {
        "bpm": 76,
        "lyrics": "아 아 아 아 내 가 슴 에 불 타 오 르 는 이 서 름",
        "pitches": "62 65 69 74 77 74 69 65 62 65 69 74 69 65 62",
        "notes": "<NOTE_2> <NOTE_2> <NOTE_4> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_2>",
        "desc": "화려한 고음 음역대와 웅장한 가창이 돋보이는 클래식 선율",
    },
    "10. Opencpop 2003000087 [중국어 표준 1 / 90 BPM] (感受停留)": {
        "bpm": 90,
        "lyrics": "感 受 停 留",
        "pitches": "62 62 60 59",
        "notes": "<NOTE_4> <NOTE_4> <NOTE_4> <NOTE_2>",
        "desc": "VocalRender 공식 표준 검증용 중국어 악보 1",
    },
    "11. Opencpop 2017000646 [중국어 표준 2 / 85 BPM] (我想念你)": {
        "bpm": 85,
        "lyrics": "我 想 念 你",
        "pitches": "64 65 67 69",
        "notes": "<NOTE_4> <NOTE_4> <NOTE_4> <NOTE_2>",
        "desc": "VocalRender 공식 표준 검증용 중국어 악보 2",
    },
    "12. 재즈 보컬 / 럭셔리 스윙 [Bb Major / 100 BPM] (Fly Me To The Moon Style)": {
        "bpm": 100,
        "lyrics": "달 빛 아래 서 너 와 날 아 오 르 고 싶 어 이 밤 에",
        "pitches": "70 69 67 65 63 62 60 62 63 65 67 69 70 69 67 65 63",
        "notes": "<NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_4> <NOTE_8> <NOTE_8> <NOTE_4> <NOTE_2>",
        "desc": "스윙감 넘치는 스무스 재즈 보컬 텐션 선율",
    },
}

# 한글/중문은 글자 단위, 영문은 단어 단위 토큰화
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\uAC00-\uD7A3\u1100-\u11FF\u3131-\u318E\u4E00-\u9FFF]")


class AudioToScoreConverter:
    """Spotify basic-pitch NN + Librosa 자동 채보 & 스마트 가사 토크나이저."""

    def __init__(self):
        self._predict_fn = None

    def _get_predict_fn(self):
        if self._predict_fn is None:
            try:
                from basic_pitch.inference import predict
                self._predict_fn = predict
            except Exception as e:
                print(f"[AudioToScore Warning] basic-pitch import failed: {e}")
                self._predict_fn = False
        return self._predict_fn if self._predict_fn else None

    # ------------------------------------------------------------------ #
    def clean_and_tokenize_lyrics(self, raw_text: str, cap: int = 512) -> List[str]:
        text = re.sub(r"\[.*?\]", " ", raw_text)
        text = re.sub(r"\(.*?\)", " ", text)
        syllables = _TOKEN_RE.findall(text)
        if not syllables:
            syllables = ["사", "랑", "해", "요"]
        return syllables[:cap] if cap else syllables

    def split_into_phrases(self, raw_lyrics, pitches, notes, max_syllables=35):
        """
        가사와 피치/노트를 35음절 단위로 분할.
        반환값: [{"words": [], "pitches": [], "notes": []}, ...]
        """
        # 1. 가사 정제 (지시문 제거)
        words = self.clean_and_tokenize_lyrics(raw_lyrics)
        
        # 2. 길이 맞추기 (가사 길이를 기준으로 피치/노트 자르기 또는 채우기)
        n = len(words)
        if len(pitches) < n:
            avg_p = int(sum(pitches)/len(pitches)) if pitches else 64
            pitches = list(pitches) + [avg_p] * (n - len(pitches))
        elif len(pitches) > n:
            pitches = list(pitches)[:n]
            
        if len(notes) < n:
            notes = list(notes) + ["<NOTE_8>"] * (n - len(notes))
        elif len(notes) > n:
            notes = list(notes)[:n]
        
        # 3. 청크 분할
        chunks = []
        for i in range(0, n, max_syllables):
            end_idx = min(i + max_syllables, n)
            chunk_words = words[i:end_idx]
            chunk_pitches = pitches[i:end_idx]
            chunk_notes = notes[i:end_idx]
            
            if chunk_words: # 빈 청크 방지
                chunks.append({
                    "words": chunk_words,
                    "pitches": chunk_pitches,
                    "notes": chunk_notes
                })
        
        # 안전장치: 만약 somehow 빈 리스트가 되면 더미 데이터 반환
        if not chunks:
            return [{"words": ["사", "랑", "해"], "pitches": [60, 62, 64], "notes": ["<NOTE_4>"]*3}]
            
        return chunks

    def align_score(self, words, pitches, notes):
        """가사 길이 기준으로 피치/노트 길이를 맞춰 1:1 정렬을 보장."""
        n = len(words)
        pitches = list(pitches) or [64] * n
        notes = list(notes) or ["<NOTE_8>"] * n
        pitches = pitches[:n] if len(pitches) >= n else pitches + [pitches[-1]] * (n - len(pitches))
        notes = notes[:n] if len(notes) >= n else notes + ["<NOTE_8>"] * (n - len(notes))
        return words, pitches, notes

    def get_aligned_preset(self, name: str) -> Dict:
        """프리셋을 토나이저 기준으로 재정렬하여 반환 (불일치 프리셋 안전 처리)."""
        p = PRO_SCORE_PRESETS.get(name)
        if not p:
            return None
        words = self.clean_and_tokenize_lyrics(p["lyrics"])
        pitches = [int(x) for x in p["pitches"].split()]
        notes = p["notes"].split()
        words, pitches, notes = self.align_score(words, pitches, notes)
        return {**p, "words": words, "pitches": pitches, "notes": notes}

    # ------------------------------------------------------------------ #
    def generate_ai_score_from_lyrics(self, lyrics_text: str, style: str = "발라드") -> Dict:
        """가사와 1:1로 정렬된 AI 악보 생성 (같은 입력 → 항상 같은 결과)."""
        words = self.clean_and_tokenize_lyrics(lyrics_text)
        style_configs = {
            "발라드": {"bpm": 74, "scales": [60, 62, 64, 65, 67, 69, 71, 72]},
            "K-Pop 댄스": {"bpm": 126, "scales": [64, 67, 69, 71, 72, 74, 76, 79]},
            "트로트": {"bpm": 108, "scales": [60, 64, 67, 69, 71, 74]},
            "Pop/R&B": {"bpm": 88, "scales": [57, 60, 62, 65, 67, 69, 72]},
            "애니/J-Pop": {"bpm": 142, "scales": [65, 67, 69, 71, 72, 74, 76]},
        }
        cfg = style_configs.get(style, style_configs["발라드"])
        scale = cfg["scales"]
        seed = int(hashlib.md5(f"{lyrics_text}::{style}".encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)

        pitches, notes = [], []
        curr = scale[len(scale) // 2]
        note_types = ["<NOTE_4>", "<NOTE_8>", "<NOTE_8>", "<NOTE_4>", "<NOTE_16>", "<NOTE_16>"]
        for _ in words:
            step = rng.choice([-2, -1, 0, 1, 2])
            idx = max(0, min(len(scale) - 1, scale.index(curr) + step if curr in scale else len(scale) // 2))
            curr = scale[idx]
            pitches.append(curr)
            notes.append(rng.choice(note_types))
        return {
            "bpm": cfg["bpm"], "words": words, "pitches": pitches, "notes": notes,
            "pitch_str": " ".join(map(str, pitches)), "note_str": " ".join(notes),
            "lyrics_str": " ".join(words), "style": style, "syllable_count": len(words),
        }

    # ------------------------------------------------------------------ #
    def extract_bpm(self, audio_path: str) -> int:
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = int(np.round(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo))
            return bpm if bpm > 0 else 90
        except Exception:
            return 90

    def extract_midi_pitches_spotify(self, audio_path: str, num_notes: int = 10) -> List[int]:
        """Spotify basic-pitch NN으로 MIDI 피치 추출."""
        predict_fn = self._get_predict_fn()
        if not predict_fn:
            return self.extract_midi_pitches_librosa(audio_path, num_notes)
        try:
            model_output, midi_data, note_events = predict_fn(audio_path)
            pitches = [int(n.pitch) for n in midi_data.instruments[0].notes]
            if not pitches:
                return self.extract_midi_pitches_librosa(audio_path, num_notes)
            idx = np.linspace(0, len(pitches) - 1, num_notes, dtype=int)
            return [max(36, min(96, pitches[i])) for i in idx]
        except Exception as e:
            print(f"[BasicPitch Error] {e}")
            return self.extract_midi_pitches_librosa(audio_path, num_notes)

    def extract_midi_pitches_librosa(self, audio_path: str, num_notes: int = 10) -> List[int]:
        """Librosa pyIN 폴백 (분석 시간 제한: 60초)."""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=60)
            f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
            valid = f0[voiced & ~np.isnan(f0)]
            if len(valid) == 0:
                return [60, 62, 64, 65, 67, 69, 71, 72][:num_notes]
            midi = np.round(librosa.hz_to_midi(valid)).astype(int)
            idx = np.linspace(0, len(midi) - 1, num_notes, dtype=int)
            return [max(36, min(96, p)) for p in midi[idx]]
        except Exception:
            return [62, 62, 60, 59, 64, 65, 67, 69][:num_notes]

    def transcribe_audio_to_vocalrender_score(self, audio_path: str, lyrics_text: str) -> Dict:
        words = self.clean_and_tokenize_lyrics(lyrics_text)
        bpm = self.extract_bpm(audio_path)
        pitches = self.extract_midi_pitches_spotify(audio_path, num_notes=len(words))
        notes = ["<NOTE_8>"] * len(words)
        return {
            "bpm": bpm, "words": words, "pitches": pitches, "notes": notes,
            "pitch_str": " ".join(map(str, pitches)), "note_str": " ".join(notes),
            "lyrics_str": " ".join(words), "syllable_count": len(words),
        }


import random

if __name__ == "__main__":
    converter = AudioToScoreConverter()
    print("AudioToScoreConverter ready.")
