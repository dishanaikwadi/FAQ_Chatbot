# faq_Chatbot.py
"""
FAQ Chatbot — full file using gTTS for Auto-Speak (stable in Streamlit)
Features:
- gTTS Auto-Speak (generates MP3 -> autoplay in browser, with st.audio fallback)
- Voice input (sounddevice -> wavio -> speech_recognition Google STT)
- TF-IDF + cosine matching for FAQs
- FAQ CRUD (View/Add/Update/Delete)
- Export FAQs CSV/XLSX, Download chat history XLSX, save chat_history.json
- Modern UI styling
- Multi-language TTS selection (English + many Indian languages)
"""

import streamlit as st
import json, os, tempfile, io, time, datetime, base64
from typing import Optional
import numpy as np

# audio and STT
import sounddevice as sd
import wavio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# gTTS for stable browser-playback TTS
from gtts import gTTS

# optional translator
try:
    from googletrans import Translator as GoogleTranslator
    translator = GoogleTranslator()
    _HAS_TRANSLATOR = True
except Exception:
    translator = None
    _HAS_TRANSLATOR = False

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# export
import pandas as pd

# download minimal NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# -------------------------
# Config & files
# -------------------------
FAQ_FILE = "faqs.json"
CHAT_HISTORY_FILE = "chat_history.json"

# supported languages for gTTS (short list + common Indian langs)
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# create default FAQs when missing
if not os.path.exists(FAQ_FILE):
    default_faqs = [
        {"question":"How do I change my order?", "answer":"Open Order History → choose order → Modify Order."},
        {"question":"How do I cancel my order?", "answer":"Go to My Orders → select order → Cancel."},
        {"question":"How long does delivery take?", "answer":"Delivery usually takes 3-7 business days depending on location."},
        {"question":"How do I return a product?", "answer":"From My Orders select the product and click Return Item then follow instructions."},
        {"question":"मैं अपना ऑर्डर कैसे रद्द कर सकता हूँ?", "answer":"'My Orders' में जाकर अपना ऑर्डर रद्द करें।"},
        {"question":"माझा ऑर्डर कसा रद्द करू शकतो?", "answer":"'My Orders' मध्ये जाऊन Cancel क्लिक करा."}
    ]
    with open(FAQ_FILE, "w", encoding="utf-8") as fh:
        json.dump(default_faqs, fh, ensure_ascii=False, indent=2)

# -------------------------
# Helpers: load/save
# -------------------------
def load_faqs():
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []

def save_faqs(faqs):
    with open(FAQ_FILE, "w", encoding="utf-8") as fh:
        json.dump(faqs, fh, ensure_ascii=False, indent=2)

# -------------------------
# Session-state defaults
# -------------------------
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

if "spoken_text" not in st.session_state:
    st.session_state.spoken_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role","text","time","audio_base64"|None}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Export utilities
# -------------------------
def export_faqs_csv_bytes(faqs):
    df = pd.DataFrame(faqs)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf

def export_faqs_excel_bytes(faqs):
    df = pd.DataFrame(faqs)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="FAQs")
    buf.seek(0)
    return buf

def export_chat_history_excel_bytes(history):
    df = pd.DataFrame(history)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ChatHistory")
    buf.seek(0)
    return buf

# -------------------------
# NLP: preprocess + matching
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)

def find_best_answer(user_question: str, faqs: list, cutoff: float = 0.2) -> Optional[dict]:
    """Return dict {'question','answer','score'} or None"""
    if not faqs or not user_question.strip():
        return None
    questions = [f.get("question","") for f in faqs]
    corpus = [preprocess_text(q) for q in questions]
    u = preprocess_text(user_question)
    try:
        vect = TfidfVectorizer().fit_transform(corpus + [u])
        cos = cosine_similarity(vect[-1], vect[:-1])
        best_idx = int(np.argmax(cos))
        best_score = float(cos[0, best_idx])
        if best_score < cutoff:
            return None
        return {"question": faqs[best_idx].get("question"), "answer": faqs[best_idx].get("answer"), "score": best_score}
    except Exception:
        return None

# -------------------------
# Safe remove and wavio wrapper
# -------------------------
def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def write_wav_file(path, fs, audio_array):
    try:
        wavio.write(path, audio_array, fs)
    except TypeError:
        wavio.write(path, audio_array, fs, sampwidth=2)

# -------------------------
# Recording & STT
# -------------------------
def record_to_wav(duration_seconds=4, fs=44100):
    try:
        placeholder = st.empty()
        placeholder.info("🔴 Recording — speak now...")
        recording = sd.rec(int(duration_seconds * fs), samplerate=fs, channels=1, dtype='int16')
        for sec in range(duration_seconds):
            placeholder.info(f"🔴 Recording — {sec+1}/{duration_seconds}s")
            time.sleep(1)
        sd.wait()
        placeholder.empty()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write_wav_file(tmp.name, fs, recording)
        return tmp.name
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def transcribe_wav_google(wav_path: str, language_code: str = "en-IN") -> str:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=language_code)
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        return f"[STT Error: {e}]"

# -------------------------
# gTTS Auto-speak (stable in Streamlit)
# -------------------------
def gtts_generate_base64_mp3(text: str, lang_code: str = "en", slow: bool = False) -> Optional[str]:
    """
    Generate an mp3 with gTTS and return base64-encoded bytes (string).
    Caller can embed the base64 into an autoplaying <audio> tag.
    """
    if not text:
        return None
    tmp_dir = tempfile.gettempdir()
    mp3_path = os.path.join(tmp_dir, f"gtts_{time.time_ns()}.mp3")
    try:
        tts = gTTS(text=text, lang=lang_code, slow=slow)
        tts.save(mp3_path)
        with open(mp3_path, "rb") as f:
            b = f.read()
        safe_remove(mp3_path)
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        safe_remove(mp3_path)
        return None

def auto_speak_gtts_play(text: str, lang_code: str = "en", slow: bool = False):
    """
    Generates base64 mp3 and attempts to autoplay via HTML + also provides st.audio fallback.
    """
    b64 = gtts_generate_base64_mp3(text, lang_code=lang_code, slow=slow)
    if not b64:
        st.warning("TTS generation failed.")
        return
    # Try autoplay via HTML (works in most browsers when user interacted with the page)
    audio_html = f"""
    <audio autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    # Also provide st.audio fallback so user can manually play if autoplay blocked
    try:
        st.audio(base64.b64decode(b64), format="audio/mp3")
    except Exception:
        pass

# -------------------------
# Translation helper (manual selected)
# -------------------------
def translate_text_if_enabled(text: str, target_lang_code: str, enable_translation: bool):
    if not text or not enable_translation:
        return text
    if not _HAS_TRANSLATOR:
        return text
    try:
        return translator.translate(text, dest=target_lang_code).text
    except Exception:
        return text

# -------------------------
# UI Styling & Layout
# -------------------------
st.set_page_config(page_title="FAQ Chatbot (gTTS Auto-Speak)", layout="wide")
st.markdown("""
<style>
.chat-box { background:#f7fbff; border-radius:12px; padding:12px; height:62vh; overflow:auto; border:1px solid #e6eef8; }
.user { background:#dafbe1; padding:10px; border-radius:10px; margin:8px 0; max-width:80%; align-self:flex-end; }
.bot { background:#ffffff; padding:10px; border-radius:10px; margin:8px 0; max-width:80%; align-self:flex-start; border:1px solid #eef6ff; }
.ts { font-size:11px; color:#666; margin-top:6px; text-align:right; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 FAQ Chatbot — gTTS Auto-Speak (Stable)")

tabs = st.tabs(["Chat", "FAQ Manager", "Settings"])

# -------------------------
# Settings tab
# -------------------------
with tabs[2]:
    st.header("Settings")
    # TTS language selector
    sel_lang_label = st.selectbox("TTS language (voice output)", list(LANGUAGE_OPTIONS.keys()), index=0)
    sel_lang_code = LANGUAGE_OPTIONS[sel_lang_label]
    st.session_state.tts_lang = sel_lang_code

    # STT language for Google Speech Recognition
    stt_lang = st.selectbox("STT language (speech recognition)", ["en-IN","hi-IN","mr-IN","ta-IN","te-IN"], index=0)
    st.session_state.stt_lang = stt_lang

    record_seconds = st.slider("Recording duration (seconds)", 2, 8, 4)
    st.session_state.record_seconds = record_seconds

    sensitivity = st.slider("Matching sensitivity (higher = stricter)", 0.05, 0.6, 0.2, step=0.01)
    st.session_state.match_cutoff = sensitivity

    enable_auto_speak = st.checkbox("Enable Auto Speak (gTTS)", value=True)
    st.session_state.auto_speak = enable_auto_speak

    enable_translate_answers = st.checkbox("Translate stored answer into selected TTS language (if translator available)", value=False)
    st.session_state.enable_translate_answers = enable_translate_answers

    st.markdown("Notes: gTTS requires internet. Autoplay may be blocked by some browsers; st.audio is provided as a fallback player.")

# -------------------------
# Chat tab
# -------------------------
with tabs[0]:
    st.subheader("Conversation")
    left, right = st.columns([3,1])

    with left:
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for msg in st.session_state.messages[-300:]:
            if msg.get("role") == "user":
                st.markdown(f"<div class='user'>{msg.get('text')}<div class='ts'>{msg.get('time')}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot'>{msg.get('text')}<div class='ts'>{msg.get('time')}</div></div>", unsafe_allow_html=True)
                # If message includes audio_base64, display player (st.audio can accept bytes)
                if msg.get("audio_base64"):
                    try:
                        st.audio(base64.b64decode(msg.get("audio_base64")), format="audio/mp3")
                    except Exception:
                        pass
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.subheader("Ask")
        # typed input widget (with a unique key)
        typed_in = st.text_input("Type your question", value=st.session_state.typed_text, key="typed_input_box")

        if st.button("🎤 Speak"):
            wav = record_to_wav(st.session_state.get("record_seconds", 4))
            if wav:
                st.audio(wav, format="audio/wav")
                transcribed = transcribe_wav_google(wav, language_code=st.session_state.get("stt_lang","en-IN"))
                safe_remove(wav)
                if transcribed == "":
                    st.warning("Could not transcribe speech. Try again or change STT language.")
                elif transcribed.startswith("[STT Error"):
                    st.error(transcribed)
                else:
                    st.session_state.spoken_text = transcribed
                    st.session_state.typed_text = transcribed
                    st.success(f"Recognized: {transcribed}")
                    try:
                        st.rerun()
                    except Exception:
                        pass

        if st.button("🔍 Get Answer"):
            widget_val = st.session_state.get("typed_input_box","").strip()
            spoken_val = st.session_state.get("spoken_text","").strip()
            if widget_val:
                q_text = widget_val
            elif spoken_val:
                q_text = spoken_val
            else:
                q_text = ""

            if not q_text:
                st.warning("Please type or speak your question first.")
            else:
                # clear spoken buffer
                st.session_state.spoken_text = ""
                st.session_state.typed_text = ""

                faqs_now = load_faqs()
                match = find_best_answer(q_text, faqs_now, cutoff=st.session_state.get("match_cutoff",0.2))
                if not match:
                    reply = "Sorry — I couldn't find a matching answer. Please add it in FAQ Manager."
                    audio_b64 = None
                else:
                    reply_text = match["answer"]
                    # If translate answers is enabled, translate stored answer to tts language code
                    if st.session_state.get("enable_translate_answers", False) and _HAS_TRANSLATOR:
                        reply_text = translate_text_if_enabled(reply_text, st.session_state.get("tts_lang","en"), True)

                    audio_b64 = None
                    if st.session_state.get("auto_speak", True):
                        slow_flag = (st.session_state.get("match_cutoff",0.2) < 0)  # not used, keep API
                        b64 = gtts_generate_base64_mp3(reply_text, lang_code=st.session_state.get("tts_lang","en"), slow=False)
                        if b64:
                            # attempt autoplay + fallback
                            auto_html = f"""
                            <audio autoplay>
                              <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                            </audio>
                            """
                            st.markdown(auto_html, unsafe_allow_html=True)
                            # st.audio fallback
                            try:
                                st.audio(base64.b64decode(b64), format="audio/mp3")
                            except Exception:
                                pass
                            audio_b64 = b64
                        else:
                            st.warning("TTS generation failed.")

                ts = datetime.datetime.now().strftime("%H:%M:%S")
                # push messages
                st.session_state.messages.append({"role":"user","text":q_text,"time":ts})
                if match:
                    st.session_state.messages.append({"role":"bot","text":reply_text,"time":ts,"audio_base64":audio_b64})
                    # record chat history row
                    st.session_state.chat_history.append({"time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user":q_text, "bot":reply_text, "stt_lang":st.session_state.get("stt_lang","en-IN"), "tts_lang":st.session_state.get("tts_lang","en")})
                else:
                    st.session_state.messages.append({"role":"bot","text":reply,"time":ts,"audio_base64":None})
                    st.session_state.chat_history.append({"time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user":q_text, "bot":reply, "stt_lang":st.session_state.get("stt_lang","en-IN"), "tts_lang":st.session_state.get("tts_lang","en")})

                try:
                    st.rerun()
                except Exception:
                    pass

        st.markdown("---")
        if st.button("🔊 Replay last reply"):
            last = next((m for m in reversed(st.session_state.messages) if m.get("role")=="bot"), None)
            if not last:
                st.info("No bot reply yet.")
            else:
                if last.get("audio_base64"):
                    try:
                        st.audio(base64.b64decode(last["audio_base64"]), format="audio/mp3")
                    except Exception:
                        # try to autoplay via HTML
                        st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{last['audio_base64']}' type='audio/mp3'></audio>", unsafe_allow_html=True)
                else:
                    st.info("No audio available for that reply.")

# -------------------------
# FAQ Manager tab
# -------------------------
with tabs[1]:
    st.header("FAQ Manager")
    faqs = load_faqs()
    action = st.selectbox("Action", ["View","Add","Update","Delete","Export"], index=0)

    if action == "View":
        if faqs:
            for i,f in enumerate(faqs,1):
                st.markdown(f"**{i}. Q:** {f.get('question')}")
                st.write(f"**A:** {f.get('answer')}")
                st.markdown("---")
        else:
            st.info("No FAQs yet. Add one below.")

    elif action == "Add":
        q_new = st.text_input("Question (new)")
        a_new = st.text_area("Answer (new)")
        if st.button("Add FAQ"):
            if q_new.strip() and a_new.strip():
                faqs.append({"question":q_new.strip(),"answer":a_new.strip()})
                save_faqs(faqs)
                st.success("FAQ added.")
                try:
                    st.rerun()
                except Exception:
                    pass
            else:
                st.warning("Both fields required.")

    elif action == "Update":
        if not faqs:
            st.info("No FAQs to update.")
        else:
            sel = st.selectbox("Choose FAQ to update", [f"{i+1}. {faqs[i]['question']}" for i in range(len(faqs))])
            idx = int(sel.split(".")[0]) - 1
            edit_q = st.text_input("Edit question", value=faqs[idx]["question"])
            edit_a = st.text_area("Edit answer", value=faqs[idx]["answer"])
            if st.button("Update FAQ"):
                if edit_q.strip() and edit_a.strip():
                    faqs[idx] = {"question":edit_q.strip(),"answer":edit_a.strip()}
                    save_faqs(faqs)
                    st.success("FAQ updated.")
                    try:
                        st.rerun()
                    except Exception:
                        pass
                else:
                    st.warning("Both fields required.")

    elif action == "Delete":
        if not faqs:
            st.info("No FAQs to delete.")
        else:
            sel = st.selectbox("Choose FAQ to delete", [f"{i+1}. {faqs[i]['question']}" for i in range(len(faqs))])
            idx = int(sel.split(".")[0]) - 1
            if st.button("Delete FAQ"):
                faqs.pop(idx)
                save_faqs(faqs)
                st.success("FAQ deleted.")
                try:
                    st.rerun()
                except Exception:
                    pass

    elif action == "Export":
        if faqs:
            st.download_button("⬇ Download FAQs (CSV)", data=export_faqs_csv_bytes(faqs), file_name="faqs.csv", mime="text/csv")
            st.download_button("⬇ Download FAQs (Excel)", data=export_faqs_excel_bytes(faqs), file_name="faqs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("No FAQs to export.")

# -------------------------
# Sidebar: Export & History
# -------------------------
with st.sidebar.expander("Export & History"):
    st.write("Export your FAQs or chat history")
    faqs_now = load_faqs()
    if faqs_now:
        st.download_button("⬇ FAQs (CSV)", data=export_faqs_csv_bytes(faqs_now), file_name="faqs.csv")
        st.download_button("⬇ FAQs (Excel)", data=export_faqs_excel_bytes(faqs_now), file_name="faqs.xlsx")
    else:
        st.info("No FAQs to export.")

    st.markdown("---")
    if st.session_state.get("chat_history"):
        st.download_button("⬇ Chat History (Excel)", data=export_chat_history_excel_bytes(st.session_state.chat_history), file_name="chat_history.xlsx")
    else:
        st.info("No chat history yet.")

    if st.button("Save chat_history.json"):
        try:
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as fh:
                json.dump(st.session_state.get("chat_history", []), fh, ensure_ascii=False, indent=2)
            st.success(f"Saved to {CHAT_HISTORY_FILE}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    if st.button("Clear saved chat_history.json"):
        try:
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
                st.success("Removed saved chat_history.json")
            else:
                st.info("No saved chat_history.json file found.")
        except Exception as e:
            st.error(f"Failed to remove file: {e}")

# Footer
st.markdown("---")
st.caption(f"FAQ file: `{FAQ_FILE}` — Session messages: {len(st.session_state.messages)} — Chat history rows: {len(st.session_state.chat_history)}")
# End of file
