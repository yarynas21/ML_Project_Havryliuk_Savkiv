import re
from typing import List, Tuple


def convert_to_ner_format(text: str, entities: List[dict], id2label: dict) -> List[Tuple[str, str]]:
    tokens = []
    current = 0

    for match in re.finditer(r'\S+', text):
        word = match.group()
        start = match.start()
        end = match.end()
        tokens.append((word, start, end))

    labeled_tokens = []
    for word, start, end in tokens:
        label = "O"
        for ent in entities:
            ent_start = ent["start"]
            ent_end = ent["end"]
            if start >= ent_start and end <= ent_end:
                label_id = int(ent["entity_group"].replace("LABEL_", ""))
                label = id2label[label_id]
                break
        labeled_tokens.append((word, label))

    final_tokens = []
    prev_label_type = None
    for word, label in labeled_tokens:
        if label.startswith("B-"):
            prev_label_type = label[2:]
            final_tokens.append((word, label))
        elif label.startswith("I-"):
            final_tokens.append((word, label))
        elif label != "O" and label.startswith("B-") == False:
            final_tokens.append((word, f"B-{label}"))
            prev_label_type = label
        elif label == "O" and prev_label_type:
            final_tokens.append((word, "O"))
            prev_label_type = None
        else:
            final_tokens.append((word, label))
            prev_label_type = None

    return final_tokens

# labeled = convert_to_ner_format(text_input, raw_results, id2label)
# for word, tag in labeled:
#     print(f"{word}\t{tag}")