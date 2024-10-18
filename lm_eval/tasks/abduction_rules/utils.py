import datasets


def doc_to_text_generative(doc):
    # doc_to_text: "We have several facts below. {{context}}\nWe have a conclusion: {{text}}}.\nQuestion: what is a missing premise?\nLet's think step by step."
    return f"We have several facts below. {doc['context']}\nWe have a conclusion: {doc['text']}.\nQuestion: what is a missing premise?"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step by step."
