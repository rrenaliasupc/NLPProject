
import whisper
from . import record

# Possible models: tiny, base, small, medium, large, turbo
# More info and requirements: https://pypi.org/project/openai-whisper/
def stt(audio_file, whisper_model="turbo", language="ca"):
    model = whisper.load_model(whisper_model)
    transcription =  model.transcribe(audio_file, language=language)["text"]
    return transcription
