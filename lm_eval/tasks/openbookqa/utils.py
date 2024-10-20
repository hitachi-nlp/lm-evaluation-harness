import datasets
from lm_eval.utils import process_choices


def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = doc['choices']['text']
        target = choices[['A', 'B', 'C', 'D', 'E'].index(doc['answerKey'])]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generative(doc):
    return f"Question: {doc['question_stem']}?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generative(doc) + "\nLet's think step by step."
