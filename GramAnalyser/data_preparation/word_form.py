# -*- coding: utf-8 -*-
# Класс Словоформы.

import numpy as np

class WordForm(object):
    def __init__(self, word: str, normal_form: str, pos: str, tag: str, vector: np.array, score: float):
        """
        :param word: сама словоформа.
        :param normal_form: лемма (= начальная форма, нормальная форма).
        :param pos: часть речи.
        :param tag: грамматическое значение.
        :param vector: вектор словоформы.
        :param score: вероятность предсказания.
        :param frequency: частота употребления в тексте.
        """
        self.word = word
        self.normal_form = normal_form
        self.pos = pos
        self.tag = tag
        self.vector = vector
        self.score = score
        self.weighted_vector = np.zeros_like(self.vector)
        self.possible_forms = []
        self.frequency = 0

    def print(self):
        return "<лемма={}; словоформа={}; часть речи={}; граммемы={}; вероятность={}; частота={}>"\
            .format(self.normal_form, self.word, self.pos, self.tag, "%0.3f" % self.score, self.frequency)

    def equals(self, other):
        return (self.normal_form, self.word, self.pos, self.tag) == \
               (other.normal_form, other.word, other.pos, other.tag)

    def hash(self):
        return hash((self.normal_form, self.word, self.pos, self.tag))
