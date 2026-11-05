from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import io

app = FastAPI(title="YAXN8 Beat Analysis Engine")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    non_silent = librosa.effects.split(y, top_db=25)
    intro_offset = float(non_silent[0][0] / sr) if len(non_silent) > 0 else 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=10)
    peak_times = librosa.frames_to_time(peaks, sr=sr).tolist()

    return {
        "status": "success",
        "bpm": round(float(np.atleast_1d(tempo)[0]), 2),
        "intro_offset": round(intro_offset, 2),
        "beat_timestamps": [round(t, 2) for t in beat_times],
        "drop_timestamps": [round(t, 2) for t in peak_times[:4]]
    }
