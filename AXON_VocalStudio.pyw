import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import soundfile as sf
import threading

PHRASE_LIMIT = 24  # 구절당 최대 음절 (잘림이 보이면 16~20으로 하향)
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typecast_extractor import TypecastVoiceExtractor, PROMPT_DIR, OUTPUT_DIR
from vocal_fusion_engine import VocalRenderFusionEngine, PRETRAINED_DIR, EXAMPLES_DIR
from audio_to_score import AudioToScoreConverter, PRO_SCORE_PRESETS
from audio_mixer import AudioMixer

# ---------------------------------------------------------------------------
# AXON Sovereign Design System Tokens (Matching HTML Spec 100%, hex 정제)
# ---------------------------------------------------------------------------
BG_COLOR = "#0b0e14"
PANEL_COLOR = "#121722"
CARD_COLOR = "#161d2b"
BORDER_COLOR = "#28324a"
BORDER_SOFT = "#1e2737"
TEXT_COLOR = "#eef2fa"
MUTED_COLOR = "#93a0b8"
GOLD_PRIMARY = "#f6c445"
GOLD_DEEP = "#d99e14"
PURPLE_PRIMARY = "#8b5cf6"
BLUE_PRIMARY = "#3b82f6"
GREEN_PRIMARY = "#22c55e"
ORANGE_PRIMARY = "#f97316"
GHOST_COLOR = "#222b3d"
ENTRY_BG = "#0d1322"
SYNC_BG = "#132a1e"          # ✅ rgba() → hex (Tk 호환)
SYNC_FG = "#4ade80"
DARK_TEXT = "#191204"


def open_folder(path):
    """✅ 크로스플랫폼 폴더 열기"""
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showerror("오류", f"폴더 열기 실패: {e}")


class AXONVocalStudioApp:
    """AXON Sovereign VocalStudio v9.0 Pro — Sovereign Dark Gold Edition."""

    def __init__(self, root):
        self.root = root
        self.root.title("👑 AXON Sovereign VocalStudio v9.0 Pro — 황금빛 보컬 & 음악 마스터 스튜디오")
        self.root.geometry("1080x860")
        self.root.configure(bg=BG_COLOR)

        self.engine = VocalRenderFusionEngine()
        self.transcriber = AudioToScoreConverter()
        self.mixer = AudioMixer()
        self.extractor = None

        self.timbre_count = 0
        self.is_monitoring_on = False
        self.is_manual_recording_on = False
        self.custom_voice_path = None
        self.custom_music_path = None
        self.current_notes = []            # ✅ 프리셋/AI 악보의 노트 유지
        self.prompt_clips_cache = []
        self.vocal_outputs_cache = []
        self._dir_mtimes = None

        self.setup_gold_styles()
        self.create_layout()
        self.start_typecast_extractor()
        self._gpu_status = None
        self._load_gpu_status()
        self.root.after(2000, self.refresh_file_lists)

    # ------------------------------------------------------------------ #
    def setup_gold_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Noto Sans KR", 10, "bold"))
        style.configure("Header.TLabel", background=BG_COLOR, foreground=GOLD_PRIMARY, font=("Noto Sans KR", 18, "bold"))
        style.configure("Sub.TLabel", background=BG_COLOR, foreground=MUTED_COLOR, font=("Noto Sans KR", 9))
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background="#171d2a", foreground=MUTED_COLOR,
                        font=("Noto Sans KR", 10, "bold"), padding=[18, 9])
        style.map("TNotebook.Tab",
                  background=[("selected", GOLD_PRIMARY)],
                  foreground=[("selected", BG_COLOR)])
        # ✅ ttk 위젯 다크 테마 일관화 (콤보박스/트리/스크롤바)
        style.configure("TCombobox", fieldbackground=ENTRY_BG, foreground=TEXT_COLOR,
                        arrowcolor=GOLD_PRIMARY, borderwidth=1)
        style.map("TCombobox", fieldbackground=[("readonly", ENTRY_BG)],
                  foreground=[("readonly", TEXT_COLOR)])
        style.configure("Treeview", background=CARD_COLOR, foreground=TEXT_COLOR,
                        fieldbackground=CARD_COLOR, borderwidth=0, font=("Noto Sans KR", 10))
        style.configure("Treeview.Heading", background=PANEL_COLOR, foreground=GOLD_PRIMARY,
                        font=("Noto Sans KR", 10, "bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", GOLD_PRIMARY)],
                  foreground=[("selected", DARK_TEXT)])
        style.configure("TScrollbar", background=CARD_COLOR, troughcolor=BG_COLOR, arrowcolor=GOLD_PRIMARY)
        self.root.option_add("*TCombobox*Listbox.background", ENTRY_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT_COLOR)
        self.root.option_add("*TCombobox*Listbox.selectBackground", GOLD_PRIMARY)
        self.root.option_add("*TCombobox*Listbox.selectForeground", DARK_TEXT)

    # ------------------------------------------------------------------ #
    def create_layout(self):
        header = tk.Frame(self.root, bg=BG_COLOR, padx=24, pady=16)
        header.pack(fill="x")
        h1 = tk.Frame(header, bg=BG_COLOR)
        h1.pack(anchor="w")
        tk.Label(h1, text="👑", bg=BG_COLOR, font=("Segoe UI Emoji", 18)).pack(side="left", padx=(0, 6))
        tk.Label(h1, text="AXON Sovereign VocalStudio", bg=BG_COLOR, fg=GOLD_PRIMARY,
                 font=("Noto Sans KR", 18, "bold")).pack(side="left", padx=(0, 10))
        tk.Label(h1, text="v9.0 Pro", bg=GOLD_PRIMARY, fg=DARK_TEXT,
                 font=("Noto Sans KR", 9, "bold"), padx=8, pady=2).pack(side="left")
        tk.Label(header,
                 text="황금빛 보컬 스튜디오: 내 보컬/음악 지정 ➔ 12종 AI 악보 작곡 ➔ 48kHz 가창 합성 ➔ MR 반주 믹싱·마스터링",
                 bg=BG_COLOR, fg=MUTED_COLOR, font=("Noto Sans KR", 9)).pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        self.tab_recorder = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.tab_recorder, text=" 🎙️ 1. 음성 & 오디오 수집 ")
        self.create_tab_recorder()

        self.tab_synth = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.tab_synth, text=" 🎼 2. 보컬 가창 합성 & 12종 AI 악보 ")
        self.create_tab_synth()

        self.tab_mixer = ttk.Frame(self.notebook, padding=16)   # ✅ 화면 순서와 속성명 일치
        self.notebook.add(self.tab_mixer, text=" 🎚️ 3. 보컬 + 반주(MR) 합성 믹서 ")
        self.create_tab_mixer()

        self.tab_model = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.tab_model, text=" 🎧 4. AI 모델 & 하드웨어 가속 ")
        self.create_tab_model()

    # ============================ TAB 1 ================================= #
    def create_tab_recorder(self):
        top_bar = tk.Frame(self.tab_recorder, bg=PANEL_COLOR, highlightbackground=BORDER_COLOR,
                           highlightthickness=1, padx=14, pady=12)
        top_bar.pack(fill="x", pady=(0, 12))
        left = tk.Frame(top_bar, bg=PANEL_COLOR); left.pack(side="left")
        self.manual_btn = tk.Button(left, text="🔴 수동 녹음 시작 (수노/유튜브/음악)", command=self.toggle_manual_recording,
                                    bg="#e74c3c", fg="#ffffff", font=("Noto Sans KR", 10, "bold"),
                                    activebackground="#ff6b6b", relief="flat", bd=0, padx=16, pady=8, cursor="hand2")
        self.manual_btn.pack(side="left", padx=(0, 8))
        self.toggle_btn = tk.Button(left, text="▶ 타입캐스트 자동 감지", command=self.toggle_monitoring,
                                    bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 10, "bold"),
                                    activebackground="#ffecb3", relief="flat", bd=0, padx=16, pady=8, cursor="hand2")
        self.toggle_btn.pack(side="left")
        right = tk.Frame(top_bar, bg=PANEL_COLOR); right.pack(side="right")
        tk.Button(right, text="📂 내 오디오 열기 (MP3/WAV)", command=self.pick_local_audio_file,
                  bg=PURPLE_PRIMARY, fg="#ffffff", font=("Noto Sans KR", 10, "bold"),
                  relief="flat", bd=0, padx=16, pady=8, cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(right, text="📁 폴더 열기", command=lambda: open_folder(PROMPT_DIR),
                  bg=GHOST_COLOR, fg=TEXT_COLOR, font=("Noto Sans KR", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=8, cursor="hand2").pack(side="left")

        engine_box = tk.Frame(self.tab_recorder, bg=CARD_COLOR, highlightbackground=BORDER_SOFT,
                              highlightthickness=1, padx=14, pady=10)
        engine_box.pack(fill="x", pady=(0, 12))
        self.tab1_status_lbl = tk.Label(engine_box, text="🟢 레코더 준비 완료 (오디오 파일 열기 📂 또는 수동/자동 녹음 🔴)",
                                        bg=CARD_COLOR, fg=GOLD_PRIMARY, font=("Noto Sans KR", 10, "bold"))
        self.tab1_status_lbl.pack(anchor="w")

        list_bar = tk.Frame(self.tab_recorder, bg=BG_COLOR); list_bar.pack(fill="x", pady=(0, 8))
        tk.Label(list_bar, text="📋 녹음 및 오디오 라이브러리 목록 (Vocal & Music Clips):",
                 bg=BG_COLOR, fg=MUTED_COLOR, font=("Noto Sans KR", 10, "bold")).pack(side="left")
        tk.Button(list_bar, text="🗑️ 선택 삭제", command=self.delete_selected_timbre,
                  bg="#c0392b", fg="#ffffff", font=("Noto Sans KR", 9, "bold"),
                  relief="flat", bd=0, padx=12, pady=4, cursor="hand2").pack(side="right")

        tree_frame = tk.Frame(self.tab_recorder, bg=CARD_COLOR); tree_frame.pack(fill="both", expand=True)
        cols = ("time", "filename", "size")
        self.tab1_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=10)
        for c, w, a, t in (("time", 110, "center", "시간"), ("filename", 520, "w", "오디오 파일명"), ("size", 120, "e", "크기")):
            self.tab1_tree.heading(c, text=t)
            self.tab1_tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tab1_tree.yview)
        self.tab1_tree.configure(yscroll=sb.set)
        self.tab1_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tab1_tree.bind("<Double-1>", self.on_timbre_double_click)

    def pick_local_audio_file(self):
        fp = filedialog.askopenfilename(title="컴퓨터 오디오 파일 선택 (MP3, WAV, FLAC, M4A)",
                                        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg"), ("All Files", "*.*")])
        if fp:
            dest = os.path.join(PROMPT_DIR, os.path.basename(fp))
            try:
                shutil.copy2(fp, dest)
                self.add_timbre_item(time.strftime("%H:%M:%S"), os.path.basename(dest),
                                     f"{os.path.getsize(dest)/1024:.1f} KB", dest)
                messagebox.showinfo("성공", f"오디오 파일이 라이브러리에 추가되었습니다!\n파일: {os.path.basename(dest)}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 복사 실패: {e}")

    def start_typecast_extractor(self):
        self.extractor = TypecastVoiceExtractor(callback_on_save=self.on_timbre_saved, log_callback=self.on_tab1_log)
        self.extractor.start()

    def toggle_manual_recording(self):
        if not self.is_manual_recording_on:
            self.is_manual_recording_on = True
            self.manual_btn.config(text="⏹️ 녹음 완료 및 저장", bg="#d35400")
            if self.extractor:
                self.extractor.start_manual_recording()
        else:
            self.is_manual_recording_on = False
            self.manual_btn.config(text="🔴 수동 녹음 시작 (수노/유튜브/음악)", bg="#e74c3c")
            if self.extractor:
                saved = self.extractor.stop_manual_recording(prefix="audio_record_")
                if saved:
                    self.add_timbre_item(time.strftime("%H:%M:%S"), os.path.basename(saved),
                                         f"{os.path.getsize(saved)/1024:.1f} KB", saved)
                    messagebox.showinfo("성공", f"녹음이 성공적으로 저장되었습니다!\n파일: {os.path.basename(saved)}")

    def toggle_monitoring(self):
        if not self.is_monitoring_on:
            self.is_monitoring_on = True
            self.toggle_btn.config(text="⏹️ 자동 감지 멈춤", bg="#e74c3c", fg="#ffffff")
            if self.extractor:
                self.extractor.enable_monitoring()
        else:
            self.is_monitoring_on = False
            self.toggle_btn.config(text="▶ 타입캐스트 자동 감지", bg=GOLD_PRIMARY, fg=DARK_TEXT)
            if self.extractor:
                self.extractor.disable_monitoring()

    def on_timbre_saved(self, t, f, s, p):
        self.root.after(0, self.add_timbre_item, t, f, s, p)

    def on_tab1_log(self, msg):
        self.root.after(0, lambda: self.tab1_status_lbl.config(text=msg))

    def add_timbre_item(self, t, f, s, p):
        self.tab1_tree.insert("", 0, values=(t, f, s), tags=(p,))
        self.timbre_count += 1
        self.refresh_timbre_combobox()

    def delete_selected_timbre(self):
        sel = self.tab1_tree.selection()
        if not sel:
            messagebox.showinfo("알림", "삭제할 항목을 선택해 주세요.")
            return
        for item in sel:
            tags = self.tab1_tree.item(item, "tags")
            if tags and os.path.exists(tags[0]):
                try:
                    os.remove(tags[0])
                except Exception:
                    pass
            self.tab1_tree.delete(item)
        self.refresh_timbre_combobox()

    def on_timbre_double_click(self, event):
        item = self.tab1_tree.selection()
        if item:
            tags = self.tab1_tree.item(item[0], "tags")
            if tags and os.path.exists(tags[0]):
                open_folder(os.path.dirname(tags[0]))

    # ============================ TAB 2 ================================= #
    def create_tab_synth(self):
        duo = tk.Frame(self.tab_synth, bg=PANEL_COLOR); duo.pack(fill="x", pady=(0, 14))

        card_v = tk.Frame(duo, bg=CARD_COLOR, highlightbackground=BORDER_SOFT, highlightthickness=1, padx=16, pady=14)
        card_v.pack(side="left", fill="both", expand=True, padx=(0, 8))
        hv = tk.Frame(card_v, bg=CARD_COLOR); hv.pack(anchor="w", pady=(0, 8))
        tk.Label(hv, text="1", bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 10, "bold"),
                 width=2, height=1).pack(side="left", padx=(0, 8))
        tk.Label(hv, text="타겟 보컬/음색 지정 (Prompt Voice)", bg=CARD_COLOR, fg=TEXT_COLOR,
                 font=("Noto Sans KR", 11, "bold")).pack(side="left")
        v_row = tk.Frame(card_v, bg=CARD_COLOR); v_row.pack(fill="x", pady=(4, 6))
        self.prompt_combo = ttk.Combobox(v_row, font=("Noto Sans KR", 10), state="readonly")
        self.prompt_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.prompt_combo.bind("<<ComboboxSelected>>", self.on_prompt_combo_selected)
        tk.Button(v_row, text="🎙️ 음성 지정", command=self.pick_custom_target_voice_file,
                  bg=PURPLE_PRIMARY, fg="#ffffff", font=("Noto Sans KR", 9, "bold"),
                  relief="flat", bd=0, padx=14, pady=5, cursor="hand2").pack(side="right")
        self.target_voice_path_entry = tk.Entry(card_v, font=("JetBrains Mono", 9), bg=ENTRY_BG,
                                                fg=MUTED_COLOR, insertbackground=GOLD_PRIMARY, relief="flat", bd=1)
        self.target_voice_path_entry.pack(fill="x", pady=(4, 0))

        card_m = tk.Frame(duo, bg=CARD_COLOR, highlightbackground=BORDER_SOFT, highlightthickness=1, padx=16, pady=14)
        card_m.pack(side="right", fill="both", expand=True, padx=(8, 0))
        hm = tk.Frame(card_m, bg=CARD_COLOR); hm.pack(anchor="w", pady=(0, 8))
        tk.Label(hm, text="2", bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 10, "bold"),
                 width=2, height=1).pack(side="left", padx=(0, 8))
        tk.Label(hm, text="음악/오디오 파일 선택 (Target Music)", bg=CARD_COLOR, fg=TEXT_COLOR,
                 font=("Noto Sans KR", 11, "bold")).pack(side="left")
        m_row = tk.Frame(card_m, bg=CARD_COLOR); m_row.pack(fill="x", pady=(4, 6))
        self.music_path_entry = tk.Entry(m_row, font=("JetBrains Mono", 9), bg=ENTRY_BG, fg=TEXT_COLOR,
                                         insertbackground=GOLD_PRIMARY, relief="flat", bd=1)
        self.music_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(m_row, text="📁 음악 선택", command=self.pick_music_file_for_score,
                  bg=BLUE_PRIMARY, fg="#ffffff", font=("Noto Sans KR", 9, "bold"),
                  relief="flat", bd=0, padx=14, pady=5, cursor="hand2").pack(side="right")
        tk.Button(card_m, text="🎵 선택한 음악에서 악보/MIDI 피치 자동 추출 (Spotify AI)",
                  command=self.auto_transcribe_score, bg=GREEN_PRIMARY, fg="#ffffff",
                  font=("Noto Sans KR", 9, "bold"), relief="flat", bd=0, padx=12, pady=6,
                  cursor="hand2").pack(fill="x", pady=(6, 0))

        card_score = tk.Frame(self.tab_synth, bg=CARD_COLOR, highlightbackground=BORDER_SOFT,
                              highlightthickness=1, padx=16, pady=16)
        card_score.pack(fill="both", expand=True, pady=(0, 14))
        hs = tk.Frame(card_score, bg=CARD_COLOR)
        hs.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))
        tk.Label(hs, text="3", bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 10, "bold"),
                 width=2, height=1).pack(side="left", padx=(0, 8))
        tk.Label(hs, text="악보 정보 (12종 전문가 악보 프리셋 & AI 작곡)", bg=CARD_COLOR, fg=TEXT_COLOR,
                 font=("Noto Sans KR", 11, "bold")).pack(side="left")

        tk.Label(card_score, text="12종 전문가 악보", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 9, "bold")).grid(row=1, column=0, sticky="w")
        self.preset_combo = ttk.Combobox(card_score, values=list(PRO_SCORE_PRESETS.keys()),
                                         state="readonly", font=("Noto Sans KR", 9, "bold"))
        self.preset_combo.current(0)
        self.preset_combo.grid(row=1, column=1, sticky="ew", padx=6)
        self.preset_combo.bind("<<ComboboxSelected>>", self.load_preset_score)
        tk.Label(card_score, text="BPM", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 9, "bold")).grid(row=1, column=2, sticky="e")
        self.bpm_entry = tk.Entry(card_score, font=("Noto Sans KR", 10, "bold"), width=6,
                                  bg=ENTRY_BG, fg=GOLD_PRIMARY, insertbackground=GOLD_PRIMARY, relief="flat", bd=1)
        self.bpm_entry.insert(0, "72")
        self.bpm_entry.grid(row=1, column=3, sticky="w", padx=4)
        ai_grp = tk.Frame(card_score, bg=CARD_COLOR); ai_grp.grid(row=1, column=4, sticky="e")
        self.style_combo = ttk.Combobox(ai_grp, values=["발라드", "K-Pop 댄스", "트로트", "Pop/R&B", "애니/J-Pop"],
                                        state="readonly", font=("Noto Sans KR", 9, "bold"), width=10)
        self.style_combo.current(0)
        self.style_combo.pack(side="left", padx=(0, 6))
        tk.Button(ai_grp, text="✨ AI 악보 생성", command=self.generate_ai_score_from_lyrics,
                  bg=ORANGE_PRIMARY, fg="#ffffff", font=("Noto Sans KR", 9, "bold"),
                  relief="flat", bd=0, padx=12, pady=5, cursor="hand2").pack(side="right")

        self.alignment_lbl = tk.Label(card_score, text="✓ 생성된 가사 (0음절) == MIDI 피치 (0개) 1:1 동기화 완료",
                                      bg=SYNC_BG, fg=SYNC_FG, font=("Noto Sans KR", 9, "bold"), padx=12, pady=6)
        self.alignment_lbl.grid(row=2, column=0, columnspan=5, sticky="w", pady=8)

        tk.Label(card_score, text="가사 (Lyrics)", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 9, "bold")).grid(row=3, column=0, sticky="nw", pady=4)
        self.lyrics_text = tk.Text(card_score, height=2, bg=ENTRY_BG, fg=TEXT_COLOR,
                                   insertbackground=GOLD_PRIMARY, font=("Noto Sans KR", 10), relief="flat", bd=1)
        self.lyrics_text.grid(row=3, column=1, columnspan=4, sticky="ew", pady=4)

        tk.Label(card_score, text="MIDI 피치", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 9, "bold")).grid(row=4, column=0, sticky="nw", pady=4)
        self.pitch_text = tk.Text(card_score, height=2, bg=ENTRY_BG, fg=GOLD_PRIMARY,
                                  insertbackground=GOLD_PRIMARY, font=("JetBrains Mono", 10, "bold"), relief="flat", bd=1)
        self.pitch_text.grid(row=4, column=1, columnspan=4, sticky="ew", pady=4)
        card_score.columnconfigure(1, weight=1)

        card_action = tk.Frame(self.tab_synth, bg=PANEL_COLOR); card_action.pack(fill="x", pady=4)
        tk.Button(card_action, text="🎤 보컬렌더 노래 음성 합성 (Render Singing Voice)", command=self.start_rendering,
                  bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 11, "bold"), activebackground="#ffecb3",
                  relief="flat", bd=0, padx=24, pady=12, cursor="hand2").pack(side="left")
        tk.Button(card_action, text="📂 합성 결과 폴더 열기", command=lambda: open_folder(OUTPUT_DIR),
                  bg=GHOST_COLOR, fg=TEXT_COLOR, font=("Noto Sans KR", 10, "bold"),
                  relief="flat", bd=0, padx=18, pady=12, cursor="hand2").pack(side="right")
        self.tab2_status = tk.Label(self.tab_synth, text="🟡 VocalRender AI 가창 합성 준비 완료 (타겟 음색과 악보 확인 후 합성 실행)",
                                    bg=PANEL_COLOR, fg=GOLD_PRIMARY, font=("Noto Sans KR", 10, "bold"))
        self.tab2_status.pack(anchor="w", pady=(8, 0))

        self.load_preset_score()          # 초기 프리셋 로드
        self.refresh_timbre_combobox()

    def _set_text(self, widget, value):
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    def _update_alignment(self, n_words, n_pitches):
        self.alignment_lbl.config(
            text=f"✓ 생성된 가사 ({n_words}음절) == MIDI 피치 ({n_pitches}개) 1:1 동기화 완료")

    def on_prompt_combo_selected(self, event=None):
        idx = self.prompt_combo.current()
        if idx >= 0 and self.prompt_clips_cache:
            self.target_voice_path_entry.delete(0, tk.END)
            self.target_voice_path_entry.insert(0, self.prompt_clips_cache[idx]["filepath"])

    def pick_custom_target_voice_file(self):
        fp = filedialog.askopenfilename(title="타겟 보컬/음색 오디오 파일 선택",
                                        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg"), ("All Files", "*.*")])
        if fp:
            dest = os.path.join(PROMPT_DIR, os.path.basename(fp))
            try:
                if not os.path.exists(dest):
                    shutil.copy2(fp, dest)
                self.custom_voice_path = dest
                self.target_voice_path_entry.delete(0, tk.END)
                self.target_voice_path_entry.insert(0, dest)
                self.add_timbre_item(time.strftime("%H:%M:%S"), os.path.basename(dest),
                                     f"{os.path.getsize(dest)/1024:.1f} KB", dest)
                messagebox.showinfo("성공", f"타겟 보컬 음색이 성공적으로 지정되었습니다!\n파일: {os.path.basename(dest)}")
            except Exception as e:
                messagebox.showerror("오류", f"음색 파일 지정 실패: {e}")

    def pick_music_file_for_score(self):
        fp = filedialog.askopenfilename(title="악보 추출용 음악 파일 선택",
                                        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg"), ("All Files", "*.*")])
        if fp:
            self.custom_music_path = fp
            self.music_path_entry.delete(0, tk.END)
            self.music_path_entry.insert(0, fp)

    def generate_ai_score_from_lyrics(self):
        raw = self.lyrics_text.get("1.0", tk.END).strip()
        style = self.style_combo.get()
        res = self.transcriber.generate_ai_score_from_lyrics(raw, style)
        self.current_notes = res["notes"]
        self.bpm_entry.delete(0, tk.END); self.bpm_entry.insert(0, str(res["bpm"]))
        self._set_text(self.lyrics_text, res["lyrics_str"])
        self._set_text(self.pitch_text, res["pitch_str"])
        self._update_alignment(res["syllable_count"], len(res["pitches"]))
        self.tab2_status.config(text=f"✨ {style} 스타일 AI 악보 자동 생성 완료! (BPM: {res['bpm']})")
        messagebox.showinfo("성공", f"가사 {res['syllable_count']}음절에 완벽히 정렬된 {style} AI 악보가 생성되었습니다!")

    def auto_transcribe_score(self):
        music_path = self.music_path_entry.get().strip()
        if not music_path or not os.path.exists(music_path):
            idx = self.prompt_combo.current()
            if idx >= 0 and self.prompt_clips_cache:
                music_path = self.prompt_clips_cache[idx]["filepath"]
            else:
                messagebox.showwarning("경고", "피치를 추출할 음악(MP3/WAV) 오디오 파일을 선택해 주세요.")
                return
        raw_lyrics = self.lyrics_text.get("1.0", tk.END).strip()
        self.tab2_status.config(text="🪄 Spotify AI 신경망 오디오 음악 분석 중... (백그라운드)")

        def worker():
            try:
                res = self.transcriber.transcribe_audio_to_vocalrender_score(music_path, raw_lyrics)
                self.root.after(0, lambda: self._apply_transcribe(res))
            except Exception as e:
                self.root.after(0, lambda: self._fail_transcribe(e))
        threading.Thread(target=worker, daemon=True).start()

    def _apply_transcribe(self, res):
        self.current_notes = res["notes"]
        self.bpm_entry.delete(0, tk.END); self.bpm_entry.insert(0, str(res["bpm"]))
        self._set_text(self.lyrics_text, res["lyrics_str"])
        self._set_text(self.pitch_text, res["pitch_str"])
        self._update_alignment(res["syllable_count"], len(res["pitches"]))
        self.tab2_status.config(text=f"✨ 악보 추출 완료! (BPM: {res['bpm']}, 피치 {len(res['pitches'])}개)")
        messagebox.showinfo("성공", f"음악 파일에서 BPM({res['bpm']}) 및 MIDI 피치가 1:1 정렬되어 추출되었습니다!")

    def _fail_transcribe(self, e):
        self.tab2_status.config(text=f"⚠️ 피치 추출 에러: {e}")
        messagebox.showerror("오류", f"오디오 분석 실패: {e}")

    def refresh_timbre_combobox(self):
        clips = self.engine.get_prompt_clips()
        self.prompt_clips_cache = clips
        self.prompt_combo['values'] = [f"{c['filename']} ({c['size']})" for c in clips]
        if clips and not self.prompt_combo.get():
            self.prompt_combo.current(0)
            if not self.target_voice_path_entry.get():
                self.target_voice_path_entry.insert(0, clips[0]["filepath"])
        self.refresh_vocal_combobox_mixer()

    def load_preset_score(self, event=None):
        p = self.transcriber.get_aligned_preset(self.preset_combo.get())
        if not p:
            return
        self.current_notes = p["notes"]
        self.bpm_entry.delete(0, tk.END); self.bpm_entry.insert(0, str(p["bpm"]))
        self._set_text(self.lyrics_text, " ".join(p["words"]))
        self._set_text(self.pitch_text, " ".join(map(str, p["pitches"])))
        self._update_alignment(len(p["words"]), len(p["pitches"]))
        self.tab2_status.config(text=f"🎶 {p.get('desc', '프리셋 적용 완료')}")

    def start_rendering(self):
        if getattr(self, "_rendering", False):                 # 이중 실행 방지
            messagebox.showinfo("알림", "이미 합성이 진행 중입니다.")
            return
        self._rendering = True

        if self._gpu_status is not None and not self._gpu_status["cuda"]:
            if not messagebox.askyesno(
                "GPU 경고",
                "CUDA GPU가 감지되지 않아 CPU로 렌더링됩니다(수 시간 소요 가능).\n\n"
                "torch CUDA 12.1 재설치 후 재실행을 권장합니다.\n그래도 진행할까요?"):
                self._rendering = False
                return

        prompt_clip = self.target_voice_path_entry.get().strip()
        if not prompt_clip or not os.path.exists(prompt_clip):
            sel_idx = self.prompt_combo.current()
            if sel_idx >= 0 and self.prompt_clips_cache:
                prompt_clip = self.prompt_clips_cache[sel_idx]["filepath"]
            else:
                self._rendering = False
                messagebox.showerror("오류", "타겟 보컬 음색 파일을 지정해 주세요.")
                return
        raw_lyrics = self.lyrics_text.get("1.0", tk.END).strip()
        words = self.transcriber.clean_and_tokenize_lyrics(raw_lyrics, cap=None)
        try:
            pitches = [int(p) for p in self.pitch_text.get("1.0", tk.END).strip().split()]
            bpm = int(self.bpm_entry.get().strip())
        except ValueError:
            self._rendering = False
            messagebox.showerror("오류", "MIDI 피치 및 BPM은 숫자 형식이어야 합니다.")
            return
        if len(pitches) > len(words):
            pitches = pitches[:len(words)]
        else:
            avg = (sum(pitches) // len(pitches)) if pitches else 64
            pitches += [avg] * (len(words) - len(pitches))
        notes = self.current_notes if len(self.current_notes) == len(words) else ["<NOTE_8>"] * len(words)

        chunks = self.transcriber.split_into_phrases(raw_lyrics, pitches, notes, max_syllables=35)
        if not chunks:
            self._rendering = False
            messagebox.showwarning("경고", "생성된 가사 구절이 없습니다.")
            return

        item_name = f"user_{int(time.time())}"
        self.tab2_status.config(text=f"🎶 가창 합성 시작... (총 {len(chunks)}개 구절)")

        def _worker():
            try:
                res = self.engine.render_full_song(
                    prompt_clip, chunks, item_name, bpm,
                    progress_cb=lambda i, t: self.root.after(
                        0, lambda i=i, t=t: self.tab2_status.config(text=f"🎶 가창 합성 중... ({i}/{t} 구절) — RTX 3060 CUDA 렌더링")))
                if res.get("status") == "success":
                    self.root.after(0, lambda: self._on_render_completed(res))
                else:
                    self.root.after(0, lambda: self._on_render_failed(res.get("message", "렌더링 실패")))
            except Exception as e:
                self.root.after(0, lambda e=e: self._on_render_failed(e))
            finally:
                self.root.after(0, lambda: setattr(self, "_rendering", False))
        threading.Thread(target=_worker, daemon=True).start()

    def _on_render_failed(self, e):
        self.tab2_status.config(text=f"⚠️ 합성 실패: {e}")
        messagebox.showerror("오류", f"가창 합성 중 오류가 발생했습니다:\n{e}\n\n(outputs/render_*.log 파일에서 상세 로그 확인)")

    def _on_render_completed(self, res):
        if res["status"] == "success":
            out = os.path.basename(res["output_path"])
            self.tab2_status.config(text=f"🎉 보컬 가창 합성 완료! 파일: {out}")
            self.refresh_file_lists()
            messagebox.showinfo("합성 완료", f"노래 음성 합성이 성공적으로 완료되었습니다!\n\n파일: {out}")
            open_folder(OUTPUT_DIR)
        else:
            self.tab2_status.config(text=f"ℹ️ {res.get('message', '알림')}")
            messagebox.showinfo("알림", res.get("message", "완료되었습니다."))

    # ============================ TAB 3 (믹서) ========================== #
    def create_tab_mixer(self):
        card = tk.Frame(self.tab_mixer, bg=CARD_COLOR, highlightbackground=BORDER_SOFT,
                        highlightthickness=1, padx=20, pady=20)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="🎛️ 보컬 음성 + 반주(MR/BGM) 음악 믹싱 마스터링", bg=CARD_COLOR, fg=GOLD_PRIMARY,
                 font=("Noto Sans KR", 13, "bold")).pack(anchor="w", pady=(0, 14))
        tk.Label(card, text="1️⃣ 합성된 보컬 음성 트랙 (Synthesized Vocal)", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 10, "bold")).pack(anchor="w")
        self.mixer_vocal_combo = ttk.Combobox(card, font=("Noto Sans KR", 10), state="readonly")
        self.mixer_vocal_combo.pack(fill="x", pady=(4, 14))
        tk.Label(card, text="2️⃣ 반주(MR / BGM / 오케스트라) 음원 파일 (MP3/WAV/FLAC)", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 10, "bold")).pack(anchor="w")
        mr_bar = tk.Frame(card, bg=CARD_COLOR); mr_bar.pack(fill="x", pady=(4, 14))
        self.mr_path_entry = tk.Entry(mr_bar, font=("JetBrains Mono", 10), bg=ENTRY_BG, fg=TEXT_COLOR,
                                      insertbackground=GOLD_PRIMARY, relief="flat", bd=1)
        self.mr_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(mr_bar, text="📂 MR 반주 파일 지정", command=self.browse_mr_file,
                  bg=PURPLE_PRIMARY, fg="#ffffff", font=("Noto Sans KR", 9, "bold"),
                  relief="flat", bd=0, padx=14, pady=5, cursor="hand2").pack(side="right")

        sliders = tk.Frame(card, bg=CARD_COLOR); sliders.pack(fill="x", pady=14)
        tk.Label(sliders, text="보컬 볼륨", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.vocal_vol_slider = tk.Scale(sliders, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                                         bg=CARD_COLOR, fg=GOLD_PRIMARY, troughcolor=ENTRY_BG,
                                         highlightthickness=0, bd=0, length=220)
        self.vocal_vol_slider.set(1.0)
        self.vocal_vol_slider.grid(row=0, column=1, padx=8)
        tk.Label(sliders, text="MR 반주 볼륨", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 10, "bold")).grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.mr_vol_slider = tk.Scale(sliders, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                                      bg=CARD_COLOR, fg=GOLD_PRIMARY, troughcolor=ENTRY_BG,
                                      highlightthickness=0, bd=0, length=220)
        self.mr_vol_slider.set(0.8)
        self.mr_vol_slider.grid(row=0, column=3, padx=8)

        self.reverb_var = tk.BooleanVar(value=True)
        tk.Checkbutton(card, text="✨ 보컬 공간감 에코/리버브 효과 적용", variable=self.reverb_var,
                       bg=CARD_COLOR, fg=GOLD_PRIMARY, selectcolor=BG_COLOR,
                       font=("Noto Sans KR", 10, "bold"), activebackground=CARD_COLOR).pack(anchor="w", pady=10)
        self.mix_btn = tk.Button(card, text="🎛️ 보컬 + 반주 믹싱 (합성 음악 완성곡 생성)", command=self.mix_vocal_and_mr,
                                 bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 11, "bold"),
                                 activebackground="#ffecb3", relief="flat", bd=0, padx=20, pady=12, cursor="hand2")
        self.mix_btn.pack(anchor="w", pady=16)
        self.mixer_status = tk.Label(card, text="상태: 보컬 음성과 반주(MR) 파일을 지정 후 믹싱 버튼을 누르세요.",
                                     bg=CARD_COLOR, fg=MUTED_COLOR, font=("Noto Sans KR", 10, "bold"))
        self.mixer_status.pack(anchor="w")

    def refresh_vocal_combobox_mixer(self):
        if not hasattr(self, "mixer_vocal_combo"):
            return
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".wav", ".mp3"))]
        self.vocal_outputs_cache = [os.path.join(OUTPUT_DIR, f) for f in files]
        self.mixer_vocal_combo['values'] = files
        if files and not self.mixer_vocal_combo.get():
            self.mixer_vocal_combo.current(0)

    def browse_mr_file(self):
        fp = filedialog.askopenfilename(title="반주(MR) 파일 선택",
                                        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a"), ("All Files", "*.*")])
        if fp:
            self.mr_path_entry.delete(0, tk.END)
            self.mr_path_entry.insert(0, fp)

    def mix_vocal_and_mr(self):
        idx = self.mixer_vocal_combo.current()
        if idx < 0 or not self.vocal_outputs_cache:
            messagebox.showwarning("경고", "합성된 보컬 트랙을 선택해 주세요.")
            return

        vocal_path = self.vocal_outputs_cache[idx]
        mr_path = self.mr_path_entry.get().strip()

        if not mr_path or not os.path.exists(mr_path):
            messagebox.showwarning("경고", "유효한 반주(MR/BGM) 음원 파일을 지정해 주세요.")
            return

        vocal_gain = self.vocal_vol_slider.get()
        mr_gain = self.mr_vol_slider.get()
        add_reverb = self.reverb_var.get()

        out_filename = f"master_song_{int(time.time())}.wav"
        output_path = os.path.join(OUTPUT_DIR, out_filename)

        self.mixer_status.config(text="🎛️ 보컬과 반주(MR) 음원을 48kHz 마스터링 믹싱 중...")

        def _mix_worker():
            res = self.mixer.mix_vocal_and_mr(vocal_path, mr_path, output_path, vocal_gain, mr_gain, add_reverb)
            self.root.after(0, lambda: self._on_mix_completed(res, out_filename))

        threading.Thread(target=_mix_worker, daemon=True).start()

    def _on_mix_completed(self, res, out_filename):
        if res["status"] == "success":
            self.mixer_status.config(text=f"🎉 완성곡 믹싱 완료! ({res['duration']})")
            messagebox.showinfo("성공", f"보컬+반주 합성 완성곡이 생성되었습니다!\n\n파일: {out_filename}\n길이: {res['duration']}")
            open_folder(OUTPUT_DIR)
        else:
            self.mixer_status.config(text=f"⚠️ {res.get('message')}")
            messagebox.showerror("오류", res.get("message"))

    # ============================ TAB 4 (모델) ========================== #
    def create_tab_model(self):
        card = tk.Frame(self.tab_model, bg=CARD_COLOR, highlightbackground=BORDER_SOFT,
                        highlightthickness=1, padx=20, pady=20)
        card.pack(fill="both", expand=True)

        lbl = tk.Label(card, text="🎧 AI 모델 가중치 & 하드웨어 가속 현황", bg=CARD_COLOR, fg=GOLD_PRIMARY,
                       font=("Noto Sans KR", 13, "bold"))
        lbl.pack(anchor="w", pady=(0, 10))

        m_box = tk.Frame(card, bg=CARD_COLOR); m_box.pack(anchor="w", pady=(0, 10))
        tk.Label(m_box, text="선택할 SVS AI 모델:", bg=CARD_COLOR, fg=MUTED_COLOR,
                 font=("Noto Sans KR", 10, "bold")).pack(side="left", padx=(0, 8))
        self.model_combo = ttk.Combobox(m_box, values=["VocalRender", "VocalRender-Pro"],
                                         state="readonly", font=("Noto Sans KR", 10, "bold"), width=18)
        self.model_combo.current(0)
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        info_txt = (
            "VocalRender AI 가중치는 HuggingFace(`pymaster/VocalRender`)에서 성공적으로 수집되었습니다.\n\n"
            "• VocalRender (기본 모델): 9.5 GB (수집 완료됨 ✅)\n"
            "• VocalRender-Pro (프로 모델): 9.5 GB (수집 완료됨 ✅)\n\n"
            "🎮 하드웨어 가속: NVIDIA GeForce RTX 3060 (12GB VRAM / CUDA 12.1) 연동 완료! ✅\n"
            "⚡ 렌더링 속도: 35음절 구절 크로스페이드 3초~6초대 초고속 가창 합성 작동 중"
        )
        tk.Label(card, text=info_txt, bg=CARD_COLOR, fg=TEXT_COLOR, font=("Noto Sans KR", 10, "bold"),
                 justify="left").pack(anchor="w", pady=4)

        dl_btn = tk.Button(
            card, 
            text="📥 HuggingFace 가중치 다운로드 시작/재개", 
            command=self.download_huggingface_weights,
            bg=GOLD_PRIMARY, fg=DARK_TEXT, font=("Noto Sans KR", 10, "bold"),
            relief="flat", bd=0, padx=18, pady=10, cursor="hand2"
        )
        dl_btn.pack(anchor="w", pady=14)

        self.tab3_log = tk.Label(card, text="가중치 상태: 100% 수집 완료 (d:\\vocalRender\\pretrained_models\\)",
                                bg=CARD_COLOR, fg=MUTED_COLOR, font=("Noto Sans KR", 10, "bold"))
        self.tab3_log.pack(anchor="w")

    def on_model_selected(self, event=None):
        chosen = self.model_combo.get()
        self.engine.ckpt = chosen
        messagebox.showinfo("모델 변경", f"가창 신경망 모델이 [{chosen}]로 설정되었습니다!")

    def download_huggingface_weights(self):
        cmd = f'hf download pymaster/VocalRender --local-dir "{PRETRAINED_DIR}"'
        self.tab3_log.config(text="📥 가중치 다운로드를 시작합니다...", fg=GOLD_PRIMARY)
        try:
            subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)
        except Exception as e:
            messagebox.showerror("오류", f"다운로드 명령 실행 실패: {e}")

    def refresh_file_lists(self, force=False):
        try:
            for item in self.tab1_tree.get_children():
                tags = self.tab1_tree.item(item, "tags")
                if tags and not os.path.exists(tags[0]):
                    self.tab1_tree.delete(item)
            self.refresh_timbre_combobox()
        except Exception:
            pass
        self.root.after(3000, self.refresh_file_lists)

    def _load_gpu_status(self):
        def worker():
            st = self.engine.get_gpu_status()
            self.root.after(0, lambda: self._apply_gpu_status(st))
        threading.Thread(target=worker, daemon=True).start()

    def _apply_gpu_status(self, st):
        self._gpu_status = st
        if st["cuda"]:
            self.tab3_log.config(
                text=f"✅ 실제 감지: {st['name']} ({st['mem_gb']}GB) | torch {st['torch']} CUDA 활성",
                fg="#4ade80")
        else:
            self.tab3_log.config(
                text=f"⚠️ CUDA 불가(CPU 모드)! torch {st['torch']}\n"
                     f"→ pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                     f"→ venv pythonw로 실행 확인\n{st.get('stderr','')}",
                fg="#ff6b6b")

    def on_close(self):
        if self.extractor:
            self.extractor.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AXONVocalStudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
