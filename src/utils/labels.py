ACTIONS = ["turn_on", "turn_off", "none"]
DEVICES = ["lights", "fan", "thermostat", "none"]
LOCATIONS = ["living_room", "kitchen", "bedroom", "bathroom", "none"]

def label_to_id(label, label_list): return label_list.index(label)
def id_to_label(i, label_list): return label_list[i]
