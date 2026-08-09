import os
import sys
import time
import wave
import numpy as np
from datetime import datetime
import pyaudiowpatch as pyaudio

PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TypecastVoiceExtractor:
    """Extracts voice/music samples with both Auto-VAD Mode and Manual Free Recording Mode (Suno/YouTube/Music)."""
    def __init__(self, callback_on_save, log_callback):
        self.callback_on_save = callback_on_save
        self.log_callback = log_callback
        
        self.p_audio = None
        self.stream = None
        self.sample_rate = 48000
        self.channels = 2
        
        self.is_monitoring = False       # Auto VAD mode toggle
        self.is_manual_recording = False # Manual free recording toggle
        self.manual_frames = []
        self.manual_start_time = None

        self.is_recording = False
        self.auto_frames = []
        self.silence_timer = None
        self.is_active = True

        self.SILENCE_THRESHOLD = 0.0025
        self.SILENCE_DURATION = 0.65
        self.MIN_AUDIO_DURATION = 0.6

    def start(self):
        try:
            self.p_audio = pyaudio.PyAudio()
            dev = self.p_audio.get_default_wasapi_loopback()
            
            self.sample_rate = int(dev['defaultSampleRate'])
            self.channels = dev['maxInputChannels']

            self.stream = self.p_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=dev['index'],
                stream_callback=self._audio_callback
            )
            
            dev_name = dev['name'].replace('[Loopback]', '').strip()
            self.log_callback(f"🟢 오디오 레코더 연동 완료 ({dev_name} 48kHz)")
            self.stream.start_stream()

        except Exception as e:
            self.log_callback(f"⚠️ 사운드 연동 오류: {e}")

    def enable_monitoring(self):
        self.is_monitoring = True
        self.is_recording = False
        self.auto_frames = []
        self.log_callback("🔴 자동 감지 작동 중! (타입캐스트 미리듣기 ▶ 클릭 시 자동 수집)")

    def disable_monitoring(self):
        self.is_monitoring = False
        self.is_recording = False
        self.auto_frames = []
        self.log_callback("⏸️ 자동 감지 멈춤")

    def start_manual_recording(self):
        self.manual_frames = []
        self.manual_start_time = time.time()
        self.is_manual_recording = True
        self.log_callback("🔴 수동 자유 녹음 진행 중... (수노/유튜브/음악 등을 재생하세요)")

    def stop_manual_recording(self, prefix="suno_music_"):
        if not self.is_manual_recording:
            return None
        self.is_manual_recording = False
        if len(self.manual_frames) > 0:
            saved_file = self._save_audio_frames(self.manual_frames, self.sample_rate, self.channels, prefix=prefix)
            self.manual_frames = []
            return saved_file
        else:
            self.log_callback("⚠️ 수집된 오디오 데이터가 없습니다.")
            return None

    def stop(self):
        self.is_active = False
        self.is_monitoring = False
        self.is_manual_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.p_audio:
            try:
                self.p_audio.terminate()
            except:
                pass

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if not self.is_active:
            return (None, pyaudio.paComplete)

        # 1. Manual Free Recording (Captures 100% of PC sound for Suno/YouTube/Music)
        if self.is_manual_recording:
            self.manual_frames.append(in_data)

        # 2. Auto-VAD Monitoring
        if self.is_monitoring:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            if len(audio_data) > 0:
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2)) / 32768.0

                if rms > self.SILENCE_THRESHOLD:
                    if not self.is_recording:
                        self.is_recording = True
                        self.auto_frames = []
                        self.log_callback("🔴 음성 재생 감지! 실시간 추출 중...")
                    self.silence_timer = None
                    self.auto_frames.append(in_data)
                else:
                    if self.is_recording:
                        self.auto_frames.append(in_data)
                        if self.silence_timer is None:
                            self.silence_timer = time.time()
                        elif time.time() - self.silence_timer >= self.SILENCE_DURATION:
                            self.is_recording = False
                            duration = (len(self.auto_frames) * frame_count) / self.sample_rate
                            
                            if duration >= self.MIN_AUDIO_DURATION and not self.is_manual_recording:
                                self._save_audio_frames(self.auto_frames, self.sample_rate, self.channels, prefix="typecast_auto_")

                            self.auto_frames = []
                            self.silence_timer = None
                            if not self.is_manual_recording:
                                self.log_callback("🔴 자동 감지 작동 중! (미리듣기 ▶ 클릭)")

        return (None, pyaudio.paContinue)

    def _save_audio_frames(self, frames, sample_rate, channels, prefix="recorded_"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}{timestamp}.wav"
        filepath = os.path.join(PROMPT_DIR, filename)

        try:
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(self.p_audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            file_size = os.path.getsize(filepath)
            size_str = f"{file_size / 1024:.1f} KB"
            time_str = datetime.now().strftime("%H:%M:%S")

            self.callback_on_save(time_str, filename, size_str, filepath)
            return filepath
        except Exception as e:
            self.log_callback(f"⚠️ 오디오 파일 저장 에러: {e}")
            return None
