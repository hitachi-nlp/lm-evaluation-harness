import random
import logging

import datasets
from lm_eval.utils import process_choices

logger = logging.getLogger(__name__)


def doc_to_target(doc) -> int:
    choices = doc['choices']['label']
    answer = doc['answerKey'].strip()
    try:
        return choices.index(answer)
    except ValueError as e:
        logger.warning('answerKey "%s" is not in choices "%s". The answer will be randomly chosen. The original error is the following:\n%s.',
                       answer,
                       str(choices),
                       str(e))
        answer = random.choice(['A', 'B', 'C', 'D', 'E'])
        return choices.index(answer)


def doc_to_choice(doc) -> int:
    return doc['choices']['text']



def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        choices = doc['choices']['text']
        if doc['answerKey'] in ['A', 'B', 'C', 'D', 'E']:
            target = choices[['A', 'B', 'C', 'D', 'E'].index(doc['answerKey'])]
        else:
            target = choices[-1]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generative(doc):
    return f"Question: {doc['question']}?\n{doc['choice_prompt']}"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generative(doc) + "\nLet's think step by step."
