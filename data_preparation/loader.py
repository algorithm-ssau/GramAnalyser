# -*- coding: utf-8 -*-
# Модуль загрузки корпусов для Обучения LSTM.

from typing import List

import nltk
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from GramAnalyser.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from GramAnalyser.data_preparation.word_dictionary import WordDictionary
from GramAnalyser.data_preparation.process_tag import convert_from_opencorpora_tag, filter_gram_tag

class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordDictionary по корпусу
    """
    def __init__(self):
        self.grammeme_vectorizer_input = GrammemeVectorizer()
        self.grammeme_vectorizer_output = GrammemeVectorizer()
        self.word_dictionary = WordDictionary()
        self.char_set = set()
        self.morph = MorphAnalyzer() # pyMorphy2
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def parse_corpora(self, file_names: List[str]):
        """
        Построить WordDictionary, GrammemeVectorizer по корпусу
        file_names: пути к файлам корпуса.
        """
        for file_name in file_names:
            with open(file_name, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        self.word_dictionary.sort()
        self.char_set = " " + "".join(self.char_set).replace(" ", "")

    def __process_line(self, line: str):
        """
        Обработка строки в корпусе с морфоразметкой.
        """
        i, text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:5]
        print(text, lemma, pos_tag)
        # Заполняем словарь.
        self.word_dictionary.add_word(text.lower())
        # набор уникальных символов
        self.char_set |= {ch for ch in text}
        # Заполняем набор возможных выходных тегов.
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        # Заполняем набор возможных входных тегов.
        for parse in self.morph.parse(text): #Получаем с помощью pyMorphy#
            pos, gram = convert_from_opencorpora_tag(self.converter, parse.tag, text)
            gram = filter_gram_tag(gram)
            self.grammeme_vectorizer_input.add_grammemes(pos, gram)
