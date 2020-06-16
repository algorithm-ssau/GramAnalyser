# -*- coding: utf-8 -*-
# Грамматический анализ с помощью LSTM. Основной класс, из которого вызывается грамматический разбор для введенного текста

from typing import List
from collections import defaultdict

import nltk
import numpy as np
from pymorphy2 import MorphAnalyzer

from GramAnalyser.data_preparation.word_form import WordForm
from GramAnalyser.config import BuildModelConfig
from GramAnalyser.settings import MODELS_PATHS

class Analyser():
    def __init__(self):
        #Подключаем конфиги
        eval_model_config_path = MODELS_PATHS["eval_model_config"]
        eval_model_weights_path = MODELS_PATHS["eval_model_weights"]
        gram_dict_input = MODELS_PATHS["gram_input"]
        gram_dict_output = MODELS_PATHS["gram_output"]
        word_dictionary = MODELS_PATHS["word_dictionary"]
        char_set_path = MODELS_PATHS["char_set"]
        build_config = MODELS_PATHS["build_config"]
        
        self.build_config = BuildModelConfig()
        self.build_config.load(build_config)

    # Грам. разбор предложения
    def analyse(self, words: List[str]) -> List[WordForm]:
        answers = []
        ###
        return answers

    # Грам. разбор выборки текста
    def analyse_sentences(self, sentences: List[List[str]], batch_size: int=64) -> List[List[WordForm]]:
        answers = []
        ###
        return answers
    
    # разбиение текста на слова и пунктуацию
    def split_text_on_words(self, text: str) -> List[str]:
        words = []
        separators = { ",", ".", ";", "-", "\"", ":", "'", "—", "(", ")", "?", "!" }
        for word in text.split(" "):
            if word == "":
                continue
            count = 0
            for s in word:
                if s in separators:
                    if count > 0:
                        words.append(word[0:count])
                    words.append(word[count])
                    word = word[count+1:len(word)]
                    count = 0
                else:
                    count += 1
            if len(word) > 0:
                words.append(word)
        return words

    # формирует словарь уникальных слов по начальной форме и считает частоту их употребления в тексте
    def get_word_dictionary_for_text(self, wordForms: List[WordForm]) -> List[WordForm]:
        uniqieWordsDictionary = []
        uniqueWords = []
        for wordForm in wordForms:
            normalForm = wordForm.normal_form
            if normalForm not in uniqueWords:
                uniqueWords.append(normalForm)
                uniqieWordsDictionary.append(wordForm)
        
        for uniqueWord in uniqieWordsDictionary:
            frequency = 0
            for wordForm in wordForms:
                if uniqueWord.normal_form == wordForm.normal_form:
                    frequency += 1
            uniqueWord.frequency = frequency
            uniqueWord.pos = self.translatePos(uniqueWord.pos)
        return uniqieWordsDictionary

    def translatePos(self, pos: str) -> str:
        if (pos == "NOUN"):
            pos = "сущ."
        elif (pos == "ADJ"):
            pos = "прил."
        elif (pos == "VERB"):
            pos = "гл."
        elif (pos == "NUM"):
            pos = "числит."
        elif (pos == "CONJ"):
            pos = "союз"           
        elif (pos == "INTJ"):
            pos = "междом."
        elif (pos == "ADP"):
            pos = "предлог"
        elif (pos == "DET"):
            pos = "местоим."
        elif (pos == "ADV"):
            pos = "нареч."           
        elif (pos == "PUNCT"):
            pos = "пункт."
        elif (pos == "PART"):
            pos = "частица"
        elif (pos == "PRON"):
            pos = "местоим."
        elif (pos == "PROPN"):
            pos = "имя собств."
        return pos