import datasets
from lm_eval.utils import process_choices


def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = ['entailiment', 'neutral', 'contradiction']
        target = choices[doc['label']]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def _doc_to_text_base(doc):
    return f"Premise: {doc['premise']}\nHypothesis: {doc['hypothesis']}\nDoes the premise entail the hypothesis, based on common sense knowledge?\n{doc['choice_prompt']}"


def doc_to_text_generative(doc):
    return _doc_to_text_base(doc) + "\nShow me only the answer."


def doc_to_text_cot_zeroshot(doc):
    return _doc_to_text_base(doc) + "\nLet's think step by step."
