import datasets
from lm_eval.utils import process_choices


def process_docs_cot_zeroshot(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = ['entailment', 'neutral']
        target = doc['gold_label']
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generation(doc):
    return f"Premise: {doc['ori_sentence']}\nHypothesis: {doc['new_sentence']}?\nQuestion: does the premise entail the hypothesis? Answer with either \"entailment\" or \"neutral\"\nAnswer:"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step by step."
