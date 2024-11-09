# Loads and parses dialogues from a text file dialogues_text.txt
# separating each utterance by __eou__ and returns a list of lists (one per dialogue).
def get_dialogs():
    with open("ijcnlp_dailydialog/dialogues_text.txt", "r", encoding="utf-8") as file:
        dialogs = file.readlines()
    parsed_dialogs = []
    for dialog in dialogs:
        d = dialog.split("__eou__")
        d = d[:-1]
        parsed_dialogs.append(d)
    return parsed_dialogs