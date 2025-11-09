# faq_Chatbot.py
import streamlit as st
import json
import os
import tempfile
import time
import datetime
import base64
import traceback

import sounddevice as sd
import wavio
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment  # optional if you later convert formats

# ---------------------------
# Configuration
# ---------------------------
FAQ_FILE = "faqs.json"
ENCODING = "utf-8"

# If ffmpeg is installed in a custom location and pydub is needed for conversions,
# uncomment and set the correct path below. If you don't use pydub conversions, leave it.
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

st.set_page_config(page_title="Smart Voice FAQ Chatbot", layout="wide")

# ---------------------------
# Utility: load/save FAQs safely (auto-convert old format)
# ---------------------------
def load_faqs():
    """Load FAQs from file, auto-convert dict -> list of dicts if needed."""
    if not os.path.exists(FAQ_FILE):
        return []
    try:
        with open(FAQ_FILE, "r", encoding=ENCODING) as f:
            data = json.load(f)
    except Exception:
        # If JSON corrupted, return empty and let user rebuild
        return []

    # If old dict format: {"q": "a", ...} convert to [{"question":q,"answer":a}, ...]
    if isinstance(data, dict):
        converted = [{"question": k, "answer": v} for k, v in data.items()]
        save_faqs(converted)
        return converted

    # If it's already a list-like structure, validate items
    if isinstance(data, list):
        valid = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                valid.append(item)
        return valid

    # Unknown structure
    return []

def save_faqs(faqs):
    try:
        with open(FAQ_FILE, "w", encoding=ENCODING) as f:
            json.dump(faqs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save FAQs: {e}")

# ---------------------------
# FAQ CRUD helpers
# ---------------------------
def add_faq(question, answer):
    faqs = load_faqs()
    faqs.append({"question": question.strip(), "answer": answer.strip()})
    save_faqs(faqs)

def update_faq(index, question, answer):
    faqs = load_faqs()
    if 0 <= index < len(faqs):
        faqs[index] = {"question": question.strip(), "answer": answer.strip()}
        save_faqs(faqs)

def delete_faq(index):
    faqs = load_faqs()
    if 0 <= index < len(faqs):
        faqs.pop(index)
        save_faqs(faqs)

# ---------------------------
# Finding an answer (fuzzy & keyword fallback)
# ---------------------------
import difflib
def find_answer(question, faqs, cutoff=0.6):
    q = question.lower().strip()
    if not faqs:
        return "Sorry — there are no FAQs yet. Add one from the sidebar."

    questions = [f["question"].lower() for f in faqs]
    best = difflib.get_close_matches(q, questions, n=1, cutoff=cutoff)
    if best:
        idx = questions.index(best[0])
        return faqs[idx]["answer"]

    # keyword fallback: match any word
    words = [w for w in q.split() if len(w) > 2]
    for i, fq in enumerate(questions):
        for w in words:
            if w in fq:
                return faqs[i]["answer"]
    return "I couldn't find an answer — please add it in the FAQ manager."

# ---------------------------
# Microphone recording (sounddevice -> wav)
# ---------------------------
def record_to_wav(duration_seconds=4, fs=44100):
    """Record audio from default microphone and return a path to a temporary WAV file."""
    try:
        placeholder = st.empty()
        # simple visual cue
        placeholder.info("🔴 Recording — speak now...")
        # Start recording (non-blocking), use dtype int16
        recording = sd.rec(int(duration_seconds * fs), samplerate=fs, channels=1, dtype='int16')
        # show countdown
        for sec in range(duration_seconds):
            placeholder.info(f"🔴 Recording — {sec+1}/{duration_seconds}s")
            time.sleep(1)
        sd.wait()
        placeholder.empty()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavio.write(tmp.name, recording, fs, sampwidth=2)
        return tmp.name
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# ---------------------------
# Transcribe wav using speech_recognition (Google Web Speech)
# ---------------------------
def transcribe_wav_file(wav_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"[STT error: {e}]"
    except Exception as e:
        return f"[STT error: {e}]"

# ---------------------------
# Text-to-speech via gTTS (returns bytes)
# ---------------------------
def synthesize_speech_bytes(text, lang="en", slow=False):
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            b = f.read()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return b
    except Exception as e:
        st.error(f"TTS failed: {e}")
        return None

# ---------------------------
# Chat message helpers (session)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # each: {"role":"user"/"bot","text":...,"time":...,"audio":bytes|None}

def add_message(role, text, audio_bytes=None):
    st.session_state["messages"].append({
        "role": role,
        "text": text,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio": audio_bytes
    })

# ---------------------------
# Sidebar - advanced settings & FAQ manager
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings & FAQ Manager")

    # Advanced settings - unique keys
    record_duration = st.slider("Recording duration (s)", min_value=2, max_value=10, value=4, key="record_duration_slider")
    stt_language = st.selectbox("STT Language (Google)", ["en-US", "hi-IN", "en-GB"], index=0, key="stt_lang")
    tts_language = st.selectbox("TTS Language", ["en", "hi", "es", "fr", "de"], index=0, key="tts_lang")
    tts_speed_choice = st.selectbox("TTS Speed", ["Normal", "Slow"], index=0, key="tts_speed")
    auto_speak = st.checkbox("Auto-speak bot replies", value=True, key="auto_speak_key")
    fuzzy_cutoff = st.slider("FAQ fuzzy cutoff", min_value=0.4, max_value=0.9, value=0.6, step=0.05, key="fuzzy_cutoff_slider")
    st.markdown("---")

    st.subheader("📚 Manage FAQs")
    faqs_list = load_faqs()

    action = st.radio("Action", ["View all", "Add", "Update", "Delete"], index=0, key="faq_action")

    if action == "View all":
        if faqs_list:
            for i, f in enumerate(faqs_list):
                st.markdown(f"**{i+1}. {f['question']}**")
                st.write(f"{f['answer']}")
                st.markdown("---")
        else:
            st.info("No FAQs yet. Use 'Add' to create FAQs.")

    elif action == "Add":
        new_q = st.text_input("Question to add", key="add_q")
        new_a = st.text_area("Answer to add", key="add_a")
        if st.button("Add FAQ", key="add_faq_btn"):
            if new_q.strip() and new_a.strip():
                add_faq(new_q, new_a)
                st.success("FAQ added.")
                st.experimental_rerun()  # tiny refresh so main UI sees updated FAQs
            else:
                st.warning("Both question and answer are required.")

    elif action == "Update":
        if faqs_list:
            sel = st.selectbox("Choose FAQ to update", [f"{i+1}. {f['question']}" for i, f in enumerate(faqs_list)], key="update_select")
            idx = int(sel.split(".")[0]) - 1
            edit_q = st.text_input("Edit question", value=faqs_list[idx]["question"], key="edit_q")
            edit_a = st.text_area("Edit answer", value=faqs_list[idx]["answer"], key="edit_a")
            if st.button("Update FAQ", key="update_faq_btn"):
                update_faq(idx, edit_q, edit_a)
                st.success("FAQ updated.")
                st.experimental_rerun()
        else:
            st.info("No FAQs to update.")

    elif action == "Delete":
        if faqs_list:
            sel = st.selectbox("Choose FAQ to delete", [f"{i+1}. {f['question']}" for i, f in enumerate(faqs_list)], key="delete_select")
            idx = int(sel.split(".")[0]) - 1
            if st.button("Delete FAQ", key="delete_faq_btn"):
                delete_faq(idx)
                st.success("FAQ deleted.")
                st.experimental_rerun()
        else:
            st.info("No FAQs to delete.")

# ---------------------------
# Main layout: left chat, right controls
# ---------------------------
left, right = st.columns([3, 1])

# CSS chat bubble styling
st.markdown("""
<style>
.user-bubble { background:#DCF8C6; padding:10px 14px; border-radius:12px; margin:8px 0; max-width:75%; align-self:flex-end;}
.bot-bubble { background:#F1F0F0; padding:10px 14px; border-radius:12px; margin:8px 0; max-width:75%; align-self:flex-start;}
.chat-box { display:flex; flex-direction:column; gap:6px; padding:12px; border:1px solid #eee; border-radius:8px; height:60vh; overflow:auto;}
.small-time { font-size:10px; color:#666; margin-top:6px; text-align:right;}
</style>
""", unsafe_allow_html=True)

with left:
    st.subheader("💬 Chat")
    # render chat area
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['text']}<div class='small-time'>{msg['time']}</div></div>", unsafe_allow_html=True)
        else:
            # embed audio player if available
            audio_html = ""
            if msg.get("audio"):
                b64 = base64.b64encode(msg["audio"]).decode("utf-8")
                audio_html = f"<div style='margin-top:6px;'><audio controls src='data:audio/mp3;base64,{b64}'></audio></div>"
            st.markdown(f"<div class='bot-bubble'>{msg['text']}{audio_html}<div class='small-time'>{msg['time']}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader("Controls")
    # text input (unique key)
    user_text = st.text_input("Type your question", key="user_text_input")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Send", key="send_btn"):
            if user_text.strip():
                add_message("user", user_text.strip())
                faqs_now = load_faqs()
                answer = find_answer(user_text.strip(), faqs_now, cutoff=fuzzy_cutoff)
                audio_bytes = None
                if auto_speak:
                    audio_bytes = synthesize_speech_bytes(answer, lang=tts_language, slow=(tts_speed_choice == "Slow"))
                add_message("bot", answer, audio_bytes=audio_bytes)
                st.rerun()
            else:
                st.warning("Type something first.")
    with colB:
        if st.button("🎤 Speak", key="speak_btn"):
            wav_path = record_to_wav(record_duration)
            if not wav_path:
                st.error("Recording failed.")
            else:
                # optional: play back
                st.audio(wav_path, format="audio/wav")
                transcribed = transcribe_wav_file(wav_path)
                # If transcribed empty or error string, handle
                if transcribed == "":
                    st.warning("Could not transcribe your speech. Try again.")
                elif transcribed.startswith("[STT error:"):
                    st.error(transcribed)
                else:
                    add_message("user", transcribed)
                    faqs_now = load_faqs()
                    answer = find_answer(transcribed, faqs_now, cutoff=fuzzy_cutoff)
                    audio_bytes = None
                    if auto_speak:
                        audio_bytes = synthesize_speech_bytes(answer, lang=tts_language, slow=(tts_speed_choice == "Slow"))
                    add_message("bot", answer, audio_bytes=audio_bytes)
                # cleanup
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
                st.rerun()

    st.markdown("---")
    if st.button("Clear chat", key="clear_chat_btn"):
        st.session_state["messages"] = []
        st.rerun()

# Footer
st.markdown("<hr><center>✨ Smart Voice FAQ Chatbot — Voice Input & Output, FAQ management</center>", unsafe_allow_html=True)
