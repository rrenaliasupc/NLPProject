from utils import stt, record

if __name__ == "__main__":
    audio_file = record.record_temp_file()
    result = stt.stt(audio_file)
    print("Transcription:", result)
