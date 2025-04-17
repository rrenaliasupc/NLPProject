from utils import stt, record, model
from utils.labels import *

if __name__ == "__main__":
    audio_file = record.record_temp_file()
    text = stt.stt(audio_file)
    print("Transcription:", text)
    outputs = model.infer(model.MODEL_DEFAULT_PATH, text)
    cute_output = {
        "action": id_to_label(outputs["action"].argmax().item(), ACTIONS),
        "device": id_to_label(outputs["device"].argmax().item(), DEVICES),
        "location": id_to_label(outputs["location"].argmax().item(), LOCATIONS)
    }
    print("Infer results:", cute_output)
