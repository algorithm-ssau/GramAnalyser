# -*- coding: utf-8 -*-
# Функции для обработки грамматических значений.

def convert_from_opencorpora_tag(to_ud, tag: str, text: str):
    """
    Конвертировать теги их формата OpenCorpora в Universal Dependencies
    
    :param to_ud: конвертер.
    :param tag: тег в OpenCorpora.
    :param text: токен.
    :return: тег в UD.
    """
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1]
    return pos, gram


def filter_gram_tag(gram: str):
    """
    Отфильтровать лишние грамматические категории и отсортировать их в составе значения.
    """
    gram = gram.strip().split("|")
    dropped = ["Animacy", "Aspect", "NumType", "Variant", "Degree", "Voice", "VerbForm", "NumForm"]
    gram = [grammem for grammem in gram if sum([drop in grammem for drop in dropped]) == 0]
    return "|".join(sorted(gram)) if gram else "_"
