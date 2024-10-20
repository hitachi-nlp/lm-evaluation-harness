# Copied from Master
import datasets

from lm_eval.utils import process_choices


def doc_to_text(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["text"] + "\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


# # https://github.com/csitfun/LogiQA2.0/blob/main/logiqa2nli/nli-prompt.py
# def doc_to_textNLI(doc):
#     maj_premise = ' '.join(list(doc['major_premise']))
#     min_premise = ' '.join(list(doc['minor_premise']))
#     hypo = doc['conclusion']
#     prompt_input = "Given the fact: " + maj_premise + ' ' + min_premise + " Does it follow that: " + hypo + " Yes or no?"
#     return prompt_input



_COT_ZEROSHOT_IDs = ['(A)', '(B)', '(C)', '(D)', '(E)']


def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc):
        if doc['answer'] not in ['a', 'b', 'c', 'd', 'e']:
            answer_idx = -1
        else:
            answer_idx = ['a', 'b', 'c', 'd', 'e'].index(doc['answer'])
        choices = doc['options']
        target = choices[answer_idx]
        doc.update(process_choices(doc, choices, target))
        return doc

    return dataset.map(_process_doc)


def doc_to_text_generative(doc) -> str:
    prompt = "Passage: " + doc["text"] + "\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(_COT_ZEROSHOT_IDs, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    return prompt


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generative(doc) + "\nLet's think step by step."
