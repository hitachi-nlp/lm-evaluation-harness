def doc_to_target(doc) -> int:
    choices = ['A', 'B', 'C', 'D', 'E']
    return choices.index(doc["label"].strip())
