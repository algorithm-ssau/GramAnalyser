# -*- coding: utf-8 -*-
# Обучение LSTM модели

import os
from typing import List

from GramAnalyser.model import LSTMModel
from GramAnalyser.config import BuildModelConfig, TrainConfig
from GramAnalyser.util.embeddings import load_embeddings
from GramAnalyser.settings import MODELS_PATHS, TEST_GOLD_JZ, TEST_GOLD_LENTA, TEST_GOLD_VK

def train(embeddings_path: str = None):
    file_names = [TEST_GOLD_JZ, TEST_GOLD_LENTA, TEST_GOLD_VK]

    train_config = TrainConfig()
    train_config.load(MODELS_PATHS["train_config"])
    if train_config.train_model_config_path is None:
        train_config.train_model_config_path = MODELS_PATHS["train_model_config"]
    if train_config.train_model_weights_path is None:
        train_config.train_model_weights_path = MODELS_PATHS["train_model_weights"]
    if train_config.main_model_config_path is None:
        train_config.main_model_config_path = MODELS_PATHS["main_model_config"]
    if train_config.main_model_weights_path is None:
        train_config.main_model_weights_path = MODELS_PATHS["main_model_weights"]
    if train_config.gram_dict_input is None:
        train_config.gram_dict_input = MODELS_PATHS["gram_input"]
    if train_config.gram_dict_output is None:
        train_config.gram_dict_output = MODELS_PATHS["gram_output"]
    if train_config.word_dictionary is None:
        train_config.word_dictionary = MODELS_PATHS["word_dictionary"]
    if train_config.char_set_path is None:
        train_config.char_set_path = MODELS_PATHS["char_set"]

    build_config = BuildModelConfig()
    build_config.load(MODELS_PATHS["build_config"])
    if build_config.char_model_weights_path is None:
        build_config.char_model_weights_path = MODELS_PATHS["char_model_weights"]
    if build_config.char_model_config_path is None:
        build_config.char_model_config_path = MODELS_PATHS["char_model_config"]

    model = LSTMModel()
    model.prepare(train_config.gram_dict_input, train_config.gram_dict_output,
                  train_config.word_dictionary, train_config.char_set_path, file_names)
    if os.path.exists(train_config.main_model_config_path) and not train_config.rewrite_model:
        model.load_train_model(build_config, train_config.train_model_config_path, train_config.train_model_weights_path)
        print(model.main_model.summary())
    else:
        embeddings = None
        if embeddings_path is not None:
            embeddings = load_embeddings(embeddings_path, model.word_dictionary, build_config.word_max_count)
        model.build(build_config, embeddings)
    model.train(file_names, train_config, build_config)
