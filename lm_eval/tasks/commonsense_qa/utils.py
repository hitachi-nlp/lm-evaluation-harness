import random
import logging

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
