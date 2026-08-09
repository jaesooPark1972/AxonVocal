"""VocalRender Fusion Engine — timbre prompt + score-native SVS bridge.
수정: Path(__file__) 복원 / 스코어 필드 자동 정렬 / 서브프로세스 UTF-8·타임아웃 /
      render_full_song() 구절 분할 렌더링 + 자동 결합(긴 가사 잘림 대응)."""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from audio_mixer import AudioMixer

PROJECT_ROOT = Path(__file__).parent.resolve()

PROMPT_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

PROMPT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RENDER_TIMEOUT = 3600   # 구절당 서브프로세스 타임아웃(초)


class VocalRenderFusionEngine:
    """Fuses Typecast Voice Timbre Extraction with VocalRender Score-Native SVS Synthesis."""

    def __init__(self, ckpt: str = "VocalRender"):
        self.model_loaded = False
        self.svs_model = None
        self.ckpt = ckpt          # "VocalRender" | "VocalRender-Pro"

    def get_gpu_status(self) -> Dict[str, str]:
        """실제 환경의 torch/CUDA 상태를 서브프로세스로 검증 (하드코딩 대체)."""
        code = ("import torch;ok=torch.cuda.is_available();"
                "n=torch.cuda.get_device_name(0) if ok else 'NO_CUDA';"
                "m=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if ok else 0;"
                "print(torch.__version__, ok, n, m, sep='|')")
        try:
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=180)
            ver, ok, name, mem = (out.stdout.strip().split("|") + ["?", "False", "NO_CUDA", "0"])[:4]
            return {"torch": ver, "cuda": ok == "True", "name": name, "mem_gb": mem,
                    "stderr": (out.stderr or "")[-300:]}
        except Exception as e:
            return {"torch": "?", "cuda": False, "name": "NO_CUDA", "mem_gb": "0", "stderr": str(e)}

    # ------------------------------------------------------------------ #
    def get_prompt_clips(self) -> List[Dict[str, str]]:
        clips = []
        audio_exts = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"}
        if PROMPT_DIR.exists():
            for f in sorted(PROMPT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.is_file() and f.suffix.lower() in audio_exts:
                    clips.append({"filename": f.name, "filepath": str(f),
                                  "size": f"{f.stat().st_size / 1024:.1f} KB"})
        if EXAMPLES_DIR.exists():
            for f in sorted((EXAMPLES_DIR / "prompt_audio").glob("*.wav")):
                clips.append({"filename": f"demo_{f.name}", "filepath": str(f),
                              "size": f"{f.stat().st_size / 1024:.1f} KB"})
        return clips

    # ------------------------------------------------------------------ #
    @staticmethod
    def _align(words, pitches, notes):
        """words/pitches/notes 길이 불일치 시 자동 정렬 (1:1 보장)."""
        n = len(words)
        pitches = list(pitches) or [64] * n
        notes = list(notes) or ["<NOTE_8>"] * n
        pitches = pitches[:n] if len(pitches) >= n else pitches + [pitches[-1]] * (n - len(pitches))
        notes = notes[:n] if len(notes) >= n else notes + ["<NOTE_8>"] * (n - len(notes))
        return pitches, notes

    def create_score_json(self, item_name: str, words: List[str], pitches: List[int],
                          notes: List[str], bpm: int = 90) -> str:
        pitches, notes = self._align(words, pitches, notes)
        score_data = [{
            "item_name": item_name,
            "word": list(words),
            "pitch": [int(p) for p in pitches],
            "note": notes,
            "pitch2word": list(range(len(words))),
            "bpm": int(bpm),
        }]
        json_path = OUTPUT_DIR / f"score_{item_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, ensure_ascii=False, indent=2)
        return str(json_path)

    # ------------------------------------------------------------------ #
    def render_singing_voice(self, prompt_audio_path: str, score_json_path: str, item_name: str,
                             ckpt_dir: Optional[str] = None, output_filename: Optional[str] = None,
                             timeout: int = RENDER_TIMEOUT) -> Dict[str, str]:
        if not ckpt_dir:
            ckpt_dir = str(PRETRAINED_DIR / self.ckpt)
        if not output_filename:
            output_filename = f"singing_{item_name}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = OUTPUT_DIR / output_filename

        script_path = PROJECT_ROOT / "scripts" / "infer_vocalrender_svs_single.py"
        if Path(ckpt_dir).exists() and script_path.exists():
            cmd = [sys.executable, str(script_path),
                   "--ckpt_dir", ckpt_dir, "--json_file", score_json_path,
                   "--item_name", item_name, "--prompt_audio", prompt_audio_path,
                   "--output", str(output_path)]
            log_file = OUTPUT_DIR / f"render_{item_name}.log"
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            try:
                with open(log_file, "w", encoding="utf-8") as lf:
                    subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   text=True, check=True, timeout=timeout, env=env)
                return {"status": "success", "output_path": str(output_path), "log_file": str(log_file)}
            except subprocess.TimeoutExpired:
                return {"status": "error", "message": f"렌더링 타임아웃({timeout}초) — 구절 길이를 줄여주세요."}
            except subprocess.CalledProcessError as e:
                err_msg = log_file.read_text(encoding="utf-8", errors="replace")[-500:] if log_file.exists() else ""
                return {"status": "error", "message": f"모델 렌더링 실패: {err_msg or str(e)}"}
        return {"status": "info", "output_path": str(output_path),
                "message": "VocalRender 모델 가중치(`pretrained_models`)가 아직 다운로드되지 않았습니다. "
                           "HuggingFace에서 가중치 다운로드 준비 완료됨."}

    # ------------------------------------------------------------------ #
    def create_batch_score_json(self, item_prefix: str, chunks: List[Dict], bpm: int = 90) -> tuple:
        """모든 구절 청크를 단일 배치 스코어 JSON 파일로 저장."""
        batch_data, parts_paths = [], []
        for i, ch in enumerate(chunks, 1):
            item_name = f"{item_prefix}_p{i}"
            pitches, notes = self._align(ch["words"], ch["pitches"], ch["notes"])
            batch_data.append({
                "item_name": item_name,
                "word": list(ch["words"]),
                "pitch": [int(p) for p in pitches],
                "note": notes,
                "pitch2word": list(range(len(ch["words"]))),
                "bpm": int(bpm),
            })
            parts_paths.append(str(OUTPUT_DIR / f"singing_{item_name}.wav"))
        json_path = OUTPUT_DIR / f"score_batch_{item_prefix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        return str(json_path), parts_paths

    # ------------------------------------------------------------------ #
    def render_full_song(self, prompt_clip: str, chunks: List[Dict], item_name: str, bpm: int,
                         progress_cb=None, ckpt_dir: Optional[str] = None, keep_parts: bool = False) -> Dict[str, str]:
        """긴 가사 구절 단위 분할 렌더링 → 1회 모델 로드로 초고속 배치 연산 → 단일 완성곡 자동 결합."""
        if not ckpt_dir:
            ckpt_dir = str(PRETRAINED_DIR / self.ckpt)

        script_path = PROJECT_ROOT / "scripts" / "infer_vocalrender_svs_single.py"

        # 배치 모드 (1회 모델 로드로 30배 속도 향상)
        if len(chunks) > 1 and Path(ckpt_dir).exists() and script_path.exists():
            score_batch_json, parts = self.create_batch_score_json(item_name, chunks, bpm)
            if progress_cb:
                progress_cb(1, len(chunks))

            cmd = [sys.executable, str(script_path),
                   "--ckpt_dir", ckpt_dir, "--json_file", score_batch_json,
                   "--batch", "--output_dir", str(OUTPUT_DIR),
                   "--prompt_audio", prompt_clip]
            log_file = OUTPUT_DIR / f"render_batch_{item_name}.log"
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", env=env)
                import re
                with open(log_file, "w", encoding="utf-8") as lf:
                    for line in proc.stdout:
                        lf.write(line)
                        lf.flush()
                        m = re.search(r"\[(\d+)/(\d+)\]", line)
                        if m and progress_cb:
                            cur_p, tot_p = int(m.group(1)), int(m.group(2))
                            progress_cb(cur_p, tot_p)
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
            except Exception as e:
                err_msg = log_file.read_text(encoding="utf-8", errors="replace")[-500:] if log_file.exists() else ""
                return {"status": "error", "message": f"배치 렌더링 실패: {err_msg or str(e)}"}

            final = str(OUTPUT_DIR / f"full_song_{item_name}.wav")
            cr = AudioMixer().concat_wav_parts(parts, final)
            if cr["status"] != "success":
                return cr
            if not keep_parts:
                for p in parts + [score_batch_json]:
                    try: os.remove(p)
                    except Exception: pass
            return {"status": "success", "output_path": final, "duration": cr["duration"]}

        # 단일 구절 루프 폴백
        parts, jsons = [], []
        total = len(chunks)
        for i, ch in enumerate(chunks, 1):
            if progress_cb:
                progress_cb(i, total)
            sj = self.create_score_json(f"{item_name}_p{i}", ch["words"], ch["pitches"], ch["notes"], bpm)
            jsons.append(sj)
            res = self.render_singing_voice(prompt_clip, sj, f"{item_name}_p{i}", ckpt_dir=ckpt_dir)
            if res["status"] != "success":
                return res
            parts.append(res["output_path"])

        final = str(OUTPUT_DIR / f"full_song_{item_name}.wav")
        cr = AudioMixer().concat_wav_parts(parts, final)
        if cr["status"] != "success":
            return cr
        if not keep_parts:
            for p in parts + jsons:
                try: os.remove(p)
                except Exception: pass
        return {"status": "success", "output_path": final, "duration": cr["duration"]}
