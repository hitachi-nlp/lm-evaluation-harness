import datasets
from lm_eval.utils import process_choices


def process_docs_cot_zeroshot(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = ['True', 'False', 'Uncertain']
        target = doc['label']
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generation(doc):
    return f"We have several facts below. {doc['premises']}\nQuestion: {doc['conclusion']}?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step"
