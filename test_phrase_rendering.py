import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typecast_extractor import PROMPT_DIR, OUTPUT_DIR
from vocal_fusion_engine import VocalRenderFusionEngine
from audio_to_score import AudioToScoreConverter

def test_full_song_rendering():
    print("[Test] Initializing VocalRender Fusion Engine & AudioToScore Converter...")
    engine = VocalRenderFusionEngine()
    transcriber = AudioToScoreConverter()

    # Get sample prompt voice clip
    clips = engine.get_prompt_clips()
    if not clips:
        print("[Test Error] No prompt voice clips found in prompt_clips dir.")
        return
    prompt_clip = clips[0]["filepath"]
    print(f"[Test] Using prompt voice: {os.path.basename(prompt_clip)}")

    # Sample long multi-phrase lyrics (3 phrases, ~55 syllables)
    raw_lyrics = (
        "그 대 내 품 에 안 겨 요 지 금 이 순 간 을 기 억 해 요\n"
        "세 상 이 변 해 도 너 의 손 을 잡 고 영 원 히 함 께 걸 어 갈 게\n"
        "빛 나 는 별 처 럼 환 하 게 비 춰 주 는 너 는 내 삶 의 빛"
    )

    words = transcriber.clean_and_tokenize_lyrics(raw_lyrics, cap=None)
    pitches = [60, 62, 64, 65, 67, 65, 64, 62, 60, 62, 64, 65, 67, 69, 67, 65, 64,
               67, 67, 69, 71, 72, 71, 69, 67, 65, 67, 69, 71, 72, 74, 72, 71, 69, 67,
               65, 69, 72, 74, 77, 76, 74, 72, 69, 67, 65, 69, 72, 74, 76, 77, 79, 77, 76]
    notes = ["<NOTE_8>"] * len(words)
    bpm = 72

    print(f"[Test] Total Words: {len(words)}, Total Pitches: {len(pitches)}")

    chunks = transcriber.split_into_phrases(raw_lyrics, pitches, notes, max_syllables=24)
    print(f"[Test] Split into {len(chunks)} phrase chunks:")
    for i, ch in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(ch['words'])} words -> {' '.join(ch['words'][:5])}...")

    item_name = f"test_bg_{int(time.time())}"
    parts = []

    for i, ch in enumerate(chunks, 1):
        print(f"\n[Test] Rendering Phrase Chunk {i}/{len(chunks)}...")
        sj = engine.create_score_json(f"{item_name}_p{i}", ch["words"], ch["pitches"], ch["notes"], bpm)
        res = engine.render_singing_voice(prompt_clip, sj, f"{item_name}_p{i}")
        if res["status"] != "success":
            print(f"[Test Error] Chunk {i} failed: {res.get('message')}")
            return
        print(f"[Test Success] Chunk {i} rendered: {os.path.basename(res['output_path'])}")
        parts.append(res["output_path"])

    # Concat WAV parts
    print("\n[Test] Concatenating phrase WAV parts into full song...")
    SR = 48000
    segs = []
    for i, p in enumerate(parts):
        data, sr = sf.read(p)
        if sr != SR:
            import librosa
            data, _ = librosa.load(p, sr=SR)
        if i:
            gl = int(SR * 0.3)
            segs.append(np.zeros((gl,) if data.ndim == 1 else (gl, data.shape[1]), dtype=np.float32))
        segs.append(data.astype(np.float32))

    out_path = os.path.join(OUTPUT_DIR, f"full_song_{item_name}.wav")
    final_data = np.concatenate(segs)
    sf.write(out_path, final_data, SR)

    # Cleanup temporary parts
    for p in parts:
        try: os.remove(p)
        except Exception: pass

    dur_sec = len(final_data) / SR
    print(f"\n[Test Complete!] 🎉 Full song successfully rendered!")
    print(f"  Final File: {out_path}")
    print(f"  Total Duration: {dur_sec:.2f} seconds")
    print(f"  File Size: {os.path.getsize(out_path) / 1024:.1f} KB")

if __name__ == "__main__":
    test_full_song_rendering()
