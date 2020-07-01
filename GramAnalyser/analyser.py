# -*- coding: utf-8 -*-
# Грамматический анализ с помощью LSTM. Основной класс, из которого вызывается грамматический разбор для введенного текста

from typing import List
from collections import defaultdict

import nltk
import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from GramAnalyser.model import LSTMModel
from GramAnalyser.data_preparation.process_tag import convert_from_opencorpora_tag, filter_gram_tag
from GramAnalyser.data_preparation.word_form import WordForm
from GramAnalyser.config import BuildModelConfig
from GramAnalyser.settings import MODELS_PATHS

class Analyser():
    def __init__(self):
        #Подключаем конфиги
        main_model_config_path = MODELS_PATHS["main_model_config"]
        main_model_weights_path = MODELS_PATHS["main_model_weights"]
        gram_dict_input = MODELS_PATHS["gram_input"]
        gram_dict_output = MODELS_PATHS["gram_output"]
        word_dictionary = MODELS_PATHS["word_dictionary"]
        char_set_path = MODELS_PATHS["char_set"]
        build_config = MODELS_PATHS["build_config"]

        self.converter = converters.converter('opencorpora-int', 'ud14')
        self.morph = MorphAnalyzer() #Pymorphy2 
        
        self.build_config = BuildModelConfig()
        self.build_config.load(build_config)

        self.model = LSTMModel()
        self.model.prepare(gram_dict_input, gram_dict_output, word_dictionary, char_set_path)
        self.model.load_main_model(self.build_config, main_model_config_path, main_model_weights_path)

    def analyse(self, words: List[str]) -> List[WordForm]:
        """
        Грам. разбор введенного текста (без разбиения на предложения)
        """               
        words_predicts = self.model.predict_gram_analysis([words], 1, self.build_config)[0]
        return self.predictionsParsing(words, words_predicts)

    def analyse_sentences(self, sentences: List[List[str]], batch_size: int=64) -> List[List[WordForm]]:
        """
        Грам. разбор выборки текста (с разбиением на отдельные предложения)
        """       
        sentences_predicts = self.model.predict_gram_analysis(sentences, batch_size, self.build_config)
        answers = []
        for words, words_predicts in zip(sentences, sentences_predicts):
            answers.append(self.predictionsParsing(words, words_predicts))
        return answers
    
    def split_text_on_words(self, text: str) -> List[str]:
        """
        Разбивает текст на слова и пунктуацию
        """
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

    def get_word_dictionary_for_text(self, wordForms: List[WordForm]) -> List[WordForm]:
        """
        Формирует словарь уникальных слов по начальной форме и рассчитывает частоту их употребления в тексте
        """
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


    def predictionsParsing(self, words: List[str], words_predicts: List[List[float]]) -> List[WordForm]:
        """
        Преобразует полученное предсказание в нормальный вид (в объект класса WordForm).
        """
        result = []
        for word, word_prob in zip(words, words_predicts[-len(words):]):
            result.append(self.wordFormBuilding(word, word_prob[1:]))
        return result

    def wordFormBuilding(self, word: str, predicts: List[float]) -> WordForm:
        """
        Собирает WordForm по номеру тега в векторизаторе и слову.
        """
        word_forms = None
        word_forms = self.morph.parse(word)

        vectorizer = self.model.grammeme_vectorizer_output
        tag_num = int(np.argmax(predicts)) # номер грамматического разбора (тега) с наибольшей вероятностью
        score = predicts[tag_num]
        full_tag = vectorizer.get_name_by_index(tag_num)
        pos, tag = full_tag.split("#")[0], full_tag.split("#")[1]
        lemma = self.getWordNormalForm(word, pos, tag, word_forms)
        vector = np.array(vectorizer.get_vector(full_tag))
        result_form = WordForm(word=word, normal_form=lemma, pos=pos, tag=tag, vector=vector, score=score)
        return result_form


    def getWordNormalForm(self, word: str, pos_tag: str, gram: str, word_forms=None):
        """
        Определяет лемму слова с помощью pyMorphy2
        """

        if word_forms is None:
            word_forms = self.morph.parse(word)
        guess = ""
        max_common_tags = 0
        for word_form in word_forms:
            word_form_pos_tag, word_form_gram = convert_from_opencorpora_tag(self.converter, word_form.tag, word)
            word_form_gram = filter_gram_tag(word_form_gram)
            common_tags_len = len(set(word_form_gram.split("|")).intersection(set(gram.split("|"))))
            if common_tags_len > max_common_tags and word_form_pos_tag == pos_tag:
                max_common_tags = common_tags_len
                guess = word_form
            if guess == "":
                guess = word_forms[0]

            lemma = guess.normal_form
        return lemma