"""Closed action vocabulary; this is not an arbitrary-language verifier."""

ACTIONS = {
    "push_left_to_right": {"label": "Pushing [something] from left to right", "zh": "从左向右推动物体"},
    "push_right_to_left": {"label": "Pushing [something] from right to left", "zh": "从右向左推动物体"},
    "cover": {"label": "Covering [something] with [something]", "zh": "覆盖物体"},
    "uncover": {"label": "Uncovering [something]", "zh": "揭开覆盖物"},
}
OPPOSITES = {
    "push_left_to_right": "push_right_to_left", "push_right_to_left": "push_left_to_right",
    "cover": "uncover", "uncover": "cover",
}


def normalize_label(value):
    return " ".join(value.replace("[", "").replace("]", "").lower().split())


def resolve_label_ids(config):
    available = {normalize_label(value): int(key) for key, value in config.id2label.items()}
    return {key: available[normalize_label(spec["label"])] for key, spec in ACTIONS.items()}
