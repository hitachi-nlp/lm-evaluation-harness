from functools import partial
from typing import List

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import get_filter
import logging

from . import extraction, selection, transformation

logger = logging.getLogger(__name__)


def build_filter_ensemble(
    filter_name: str, components: List[List[str]]
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            kwargs = {}
        # create a filter given its name in the registry
        if isinstance(function, str):
            filter_ = get_filter(function)
        else:
            filter_ = function
        f = partial(filter_, **kwargs)
        # add the filter as a pipeline step
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)
