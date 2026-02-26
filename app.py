import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import io
from scipy.signal import butter, lfilter

st.set_page_config(page_title="AI Music Remix & Mood Generator", layout="wide")

st.title("🎵 AI Music Remix & Mood Generator")
st.write("Generate music by mood or remix uploaded audio using AI-based transformations.")

# ===============================
# MOOD MUSIC GENERATOR
# ===============================

def generate_tone(frequency, duration=2, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    return tone

def generate_mood_music(mood):
    sr = 22050
    if mood == "Happy":
        freqs = [440, 550, 660]
    elif mood == "Sad":
        freqs = [220, 261, 329]
    elif mood == "Energetic":
        freqs = [660, 880, 990]
    elif mood == "Calm":
        freqs = [196, 247, 294]
    else:
        freqs = [440]

    music = np.concatenate([generate_tone(f) for f in freqs])
    return music, sr

# ===============================
# REMIX FUNCTIONS
# ===============================

def change_pitch(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def change_speed(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def apply_lowpass_filter(data, cutoff=3000, fs=22050):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# ===============================
# UI
# ===============================

option = st.sidebar.selectbox("Choose Feature", ["Mood Generator", "Remix Uploaded Song"])

if option == "Mood Generator":
    st.subheader("🎶 Generate Music by Mood")
    mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])

    if st.button("Generate Music"):
        music, sr = generate_mood_music(mood)

        file_path = "mood_music.wav"
        sf.write(file_path, music, sr)

        st.audio(file_path)
        st.success("Music Generated Successfully!")

        with open(file_path, "rb") as f:
            st.download_button("Download Music", f, file_name="mood_music.wav")

elif option == "Remix Uploaded Song":
    st.subheader("🎧 Upload and Remix Song")
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=None)

        pitch = st.slider("Pitch Shift (semitones)", -5, 5, 0)
        speed = st.slider("Speed Change", 0.5, 2.0, 1.0)
        filter_option = st.checkbox("Apply Low Pass Filter")

        if st.button("Apply Remix"):
            y_mod = change_pitch(y, sr, pitch)
            y_mod = change_speed(y_mod, speed)

            if filter_option:
                y_mod = apply_lowpass_filter(y_mod, cutoff=3000, fs=sr)

            remix_file = "remix_output.wav"
            sf.write(remix_file, y_mod, sr)

            st.audio(remix_file)
            st.success("Remix Created Successfully!")

            with open(remix_file, "rb") as f:
                st.download_button("Download Remix", f, file_name="remix_output.wav")