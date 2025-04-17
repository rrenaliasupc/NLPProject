import sys
from utils import stt, record, model

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == '--text':
        text = sys.argv[2]
    else:
        audio_file = record.record_temp_file()
        text = stt.stt(audio_file)
        print("Transcription:", text)

    print()

    outputs = model.infer(model.MODEL_DEFAULT_PATH, text)
    print("Infer results:", outputs)
