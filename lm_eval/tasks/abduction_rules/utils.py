import datasets
import re
import random
import logging

logger = logging.getLogger(__name__)


# from lm_eval.utils import process_choices


# def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:
# 
#     def _process_doc(doc):
#         choices = doc['answers']
#         target = choices[doc['label']]
#         doc.update(process_choices(doc, choices, target))
#         return doc
# 
#     return dataset.map(_process_doc)
# 
# 
# def doc_to_text_generative(doc):
#     return f"We have several facts below. {doc['context']}\nQuestion: {doc['question']}?\n{doc['choice_prompt']}"
# 
# 
# def doc_to_text_cot_zeroshot(doc):
#     return doc_to_text_generative(doc) + "\nLet's think step by step."


def doc_to_choice(doc):
    num_choice = 5

    target = doc['label']

    if target.find(' is ') < 0:
        # the target is something like "Fiona chases mouse."
        # we omit such cases as the implementation for verbs is (only) a bit complex.
        # This should be no problem, as such case is very rare.
        logger.info(f"Skipping creating choices for the following target, as it includes verb."
                    f" This should be no problem, as such case is very rare: target='{target}'")

        return [target]

    # target is like "Fiona is big."
    target_subj, target_adj = target.rstrip('.').split(' is ')

    context = doc['context']
    all_other_adjs = {
        adj for adj in re.findall(r'\b(?:is|are)\s([a-zA-Z]+)(?=[,\.\s])', context)
        if adj != target_adj
    }
    all_other_adjs = list(all_other_adjs)

    random_state = random.getstate()
    random.seed(42)

    distractor_adjs = random.sample(all_other_adjs, num_choice - 1)
    distractor_choices = [f"{target_subj} is {adj}." for adj in distractor_adjs]

    target_idx = random.randint(0, num_choice - 1)
    choices = distractor_choices[:target_idx] + [target] + distractor_choices[target_idx:]

    random.setstate(random_state)

    return choices


def doc_to_text_generative(doc):
    # doc_to_text: "We have several facts below. {{context}}\nWe have a conclusion: {{text}}}.\nQuestion: what is a missing premise?\nLet's think step by step."
    return f"We have several facts below. {doc['context']}\nWe have a conclusion: {doc['text']}.\nQuestion: what is a missing premise?"


def doc_to_text_cot_zeroshot(doc):
    return doc_to_text_generation(doc) + "\nLet's think step by step."
