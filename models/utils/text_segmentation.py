from typing import Tuple, List
import re

separator = '\n'


def get_paragraphs_of(text: str) -> List[str]:
    paragraphs = re.split(separator, text)
    return paragraphs


def get_start_indices(style_change_indices, paragraphs):
    indices = style_change_indices
    breaches = []
    if len(indices):
        text = []
        length = 0
        for i in range(len(indices) - 1):
            text.append('\n'.join([*paragraphs[indices[i]:indices[i + 1]], '']))
            length += len(text[-1])
            breaches.append(length)
    return breaches
