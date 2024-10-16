import re

import datasets
from lm_eval.utils import process_choices


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)



def process_docs_cot_zeroshot(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        ids = ['(A)', '(B)', '(C)', '(D)', '(E)']
        choices = [preprocess(ending) for ending in doc["endings"]]
        answer_id = ids[int(doc["label"])]

        doc['choices'] = choices
        doc['answer_id'] = answer_id
        doc['choice_prompt'] = ' '.join([id + ' ' + choice for id, choice in zip(ids, choices)])

        doc['query'] = preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        doc['gold'] = int(doc["label"])

        return doc

    return dataset.map(_process_doc)


def process_docs_cot_zeroshot(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        doc['query'] = preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        doc['gold'] = int(doc["label"])

        choices = [preprocess(ending) for ending in doc["endings"]]
        target = choices[int(doc['label'])]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generation(doc):
    return f"{doc['query']}\nWhich is the best choice to follow after this text? ?\n{doc['choice_prompt']}\n"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step by step."
