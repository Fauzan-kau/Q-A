import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from playsound3 import playsound


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=None) as source:
        print("Please ask your question:")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)  # type: ignore
        return text
    except Exception as e:
        return f"Sorry, could not understand audio. Error: {str(e)}"


def text_to_speech(text):
    # Create a temp file path (without locking it)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name

    try:
        # Save TTS output
        tts = gTTS(text=text, lang='en')
        tts.save(temp_path)

        # Play the audio
        playsound(temp_path)

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
