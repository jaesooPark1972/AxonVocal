"""Mixes Synthesized Vocal + MR/BGM into mastered songs, and concatenates phrase renders."""
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List


class AudioMixer:

    def mix_vocal_and_mr(self, vocal_path: str, mr_path: str, output_path: str,
                         vocal_gain: float = 1.0, mr_gain: float = 0.8,
                         add_reverb: bool = False) -> Dict[str, str]:
        try:
            target_sr = 48000
            y_vocal, _ = librosa.load(vocal_path, sr=target_sr, mono=False)
            y_mr, _ = librosa.load(mr_path, sr=target_sr, mono=False)
            if y_vocal.ndim == 1: y_vocal = np.vstack([y_vocal, y_vocal])
            if y_mr.ndim == 1:    y_mr = np.vstack([y_mr, y_mr])

            # DC 오프셋 제거(클릭/웅웅거림 방지)
            y_vocal = y_vocal - y_vocal.mean(axis=1, keepdims=True)
            y_mr = y_mr - y_mr.mean(axis=1, keepdims=True)

            y_vocal = y_vocal * vocal_gain
            y_mr = y_mr * mr_gain

            max_len = max(y_vocal.shape[1], y_mr.shape[1])
            y_vocal = np.pad(y_vocal, ((0, 0), (0, max_len - y_vocal.shape[1])))[:, :max_len]
            y_mr = np.pad(y_mr, ((0, 0), (0, max_len - y_mr.shape[1])))[:, :max_len]

            if add_reverb:
                delay = int(target_sr * 0.04)
                vocal_reverb = np.zeros_like(y_vocal)
                vocal_reverb[:, delay:] = y_vocal[:, :-delay] * 0.3
                y_vocal = y_vocal + vocal_reverb

            mixed = y_vocal + y_mr

            # 페이드 인/아웃 (시작/끝 클릭 노이즈 방지)
            fi, fo = int(target_sr * 0.02), int(target_sr * 0.5)
            mixed[:, :fi] *= np.linspace(0.0, 1.0, fi)
            mixed[:, -fo:] *= np.linspace(1.0, 0.0, fo)

            max_val = np.max(np.abs(mixed))
            if max_val > 0.98:
                mixed = (mixed / max_val) * 0.98

            sf.write(output_path, mixed.T, target_sr)
            return {"status": "success", "output_path": output_path,
                    "duration": f"{max_len / target_sr:.1f}초", "sample_rate": f"{target_sr} Hz"}
        except Exception as e:
            return {"status": "error", "message": f"믹싱 작업 에러: {e}"}

    def concat_wav_parts(self, paths: List[str], output_path: str, gap_sec: float = 0.3) -> Dict[str, str]:
        """구절 렌더링 결과물을 50ms 크로스페이드로 부드럽게 결합 (이음매 클릭 소멸)."""
        try:
            if not paths:
                return {"status": "error", "message": "결합할 오디오가 없습니다."}
            target_sr = 48000
            fade_samples = int(target_sr * 0.05)  # 50ms 페이드

            parts = []
            for p in paths:
                y, _ = librosa.load(p, sr=target_sr, mono=False)
                if y.ndim == 1:
                    y = np.vstack([y, y])
                parts.append(y.astype(np.float32))

            overlap = min(fade_samples, min(p.shape[1] for p in parts) // 4)
            mixed = parts[0]

            for i in range(1, len(parts)):
                next_part = parts[i]
                fade_out = np.linspace(1.0, 0.0, overlap)
                fade_in = np.linspace(0.0, 1.0, overlap)

                mixed[:, -overlap:] *= fade_out
                next_part[:, :overlap] *= fade_in

                mixed = np.hstack([mixed[:, :-overlap], mixed[:, -overlap:] + next_part[:, :overlap], next_part[:, overlap:]])

            sf.write(output_path, mixed.T, target_sr)
            return {"status": "success", "output_path": str(output_path),
                    "duration": f"{mixed.shape[1] / target_sr:.1f}초"}
        except Exception as e:
            return {"status": "error", "message": f"오디오 결합 실패: {e}"}


if __name__ == "__main__":
    mixer = AudioMixer()
    print("AudioMixer module ready.")
