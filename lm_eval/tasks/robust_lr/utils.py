import datasets

from lm_eval.utils import process_choices

def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = ['entailment', 'contradiction', 'neutral']
        target = doc['label']
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generative(doc):
    return f"We have several facts below. {doc['context']}\nQuestion: {doc['statement']}?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generative(doc) + "\nLet's think step by step."
