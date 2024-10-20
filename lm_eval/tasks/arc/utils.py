import datasets

from lm_eval.utils import process_choices

def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        if doc['answerKey'] not in ['A', 'B', 'C', 'D', 'E']:
            answer_idx = -1
        else:
            answer_idx = ['A', 'B', 'C', 'D', 'E'].index(doc['answerKey'])

        choices = doc['choices']['text']
        target = choices[answer_idx]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generative(doc):
    return f"Question: {doc['question']}?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generative(doc) + "\nLet's think step by step."
