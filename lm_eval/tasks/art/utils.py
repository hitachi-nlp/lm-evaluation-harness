import datasets

from lm_eval.utils import process_choices

def process_docs_cot_zeroshot(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [doc['hypothesis_1'], doc['hypothesis_2']]
        target = choices[doc['label'] - 1]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generation(doc):
    return f"We have several facts below. {doc['observation_1']} {doc['observation_2']}\nwhich hypothesis is more plausible?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step by step."
