# -*- coding: utf-8 -*-
# LSTM-модель, предсказывающая PoS и грамм.разбор текста

from typing import List, Tuple
import os

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from keras import backend as K

from GramAnalyser.training_set_generator import TrainingSetGenerator
from GramAnalyser.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from GramAnalyser.data_preparation.word_dictionary import WordDictionary
from GramAnalyser.data_preparation.loader import Loader
from GramAnalyser.char_embedding_model import build_dense_chars_layer, get_char_model
from GramAnalyser.config import BuildModelConfig, TrainConfig

class LSTMModel:
    def __init__(self):
        self.morph = MorphAnalyzer() # использ. pyMorphy2 
        self.converter = converters.converter('opencorpora-int', 'ud14')
        self.grammeme_vectorizer_input = GrammemeVectorizer()
        self.grammeme_vectorizer_output = GrammemeVectorizer()
        self.word_dictionary = WordDictionary()
        self.char_set = ""
        self.train_model = None 
        self.main_model = None

    def prepare(self, gram_dump_path_input: str, gram_dump_path_output: str,
                word_dictionary_dump_path: str, char_set_dump_path: str,
                file_names: List[str] = None) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря слов по корпусу.
        """
        if os.path.exists(gram_dump_path_input):
            self.grammeme_vectorizer_input.load(gram_dump_path_input)
        if os.path.exists(gram_dump_path_output):
            self.grammeme_vectorizer_output.load(gram_dump_path_output)
        if os.path.exists(word_dictionary_dump_path):
            self.word_dictionary.load(word_dictionary_dump_path)
        if os.path.exists(char_set_dump_path):
            with open(char_set_dump_path, 'r', encoding='utf-8') as f:
                self.char_set = f.read().rstrip()
        if self.grammeme_vectorizer_input.is_empty() or \
                self.grammeme_vectorizer_output.is_empty() or \
                self.word_dictionary.is_empty() or \
                not self.char_set:
            loader = Loader()
            loader.parse_corpora(file_names)

            self.grammeme_vectorizer_input = loader.grammeme_vectorizer_input
            self.grammeme_vectorizer_input.save(gram_dump_path_input)
            self.grammeme_vectorizer_output = loader.grammeme_vectorizer_output
            self.grammeme_vectorizer_output.save(gram_dump_path_output)
            self.word_dictionary = loader.word_dictionary
            self.word_dictionary.save(word_dictionary_dump_path)
            self.char_set = loader.char_set
            with open(char_set_dump_path, 'w', encoding='utf-8') as f:
                f.write(self.char_set)

    def save_model(self, model_config_path: str, model_weights_path: str,
             main_model_config_path: str, main_model_weights_path: str):
        if self.main_model is not None:
            with open(main_model_config_path, "w", encoding='utf-8') as f:
                f.write(self.main_model.to_yaml())
            self.main_model.save_weights(main_model_weights_path)
        if self.train_model is not None:
            with open(model_config_path, "w", encoding='utf-8') as f:
                f.write(self.train_model.to_yaml())
            self.train_model.save_weights(model_weights_path)