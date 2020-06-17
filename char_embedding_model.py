# -*- coding: utf-8 -*-
# LSTM для символьных эмбэддингов

import os
import copy
from typing import Tuple

import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, TimeDistributed
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

from GramAnalyser.data_preparation.word_dictionary import WordDictionary

def build_dense_chars_layer(max_word_length, char_vocab_size, char_emb_dim,
                            hidden_dim, output_dim, dropout):
    chars_embedding_layer = Embedding(char_vocab_size, char_emb_dim, name='chars_embeddings')
    chars_dense_1 = Dense(hidden_dim, activation='relu')
    chars_dense_2 = Dense(output_dim)

    def dense_layer(inp):
        if len(K.int_shape(inp)) == 3:
            chars_embedding = TimeDistributed(chars_embedding_layer)(inp)
            chars_embedding = Reshape((-1, char_emb_dim * max_word_length))(chars_embedding)
        elif len(K.int_shape(inp)) == 2:
            chars_embedding = chars_embedding_layer(inp)
            chars_embedding = Reshape((char_emb_dim * max_word_length,))(chars_embedding)
        else:
            assert False
        chars_embedding = Dropout(dropout)(chars_embedding)
        chars_embedding = Dropout(dropout)(chars_dense_1(chars_embedding))
        chars_embedding = Dropout(dropout)(chars_dense_2(chars_embedding))
        return chars_embedding

    return dense_layer


class CharEmbeddingsModel:
    def __init__(self):
        self.model = None  # type: Model
        self.char_layer = None

    def save(self, model_config_path: str, model_weights_path: str):
        with open(model_config_path, "w", encoding='utf-8') as f:
            f.write(self.model.to_yaml())
        self.model.save_weights(model_weights_path)

    def load(self, model_config_path: str, model_weights_path: str) -> None:
        with open(model_config_path, "r", encoding='utf-8') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(model_weights_path)
        self.char_layer = TimeDistributed(Model(self.model.input_layers[0].output, self.model.layers[-2].input))

    def build(self,
              char_layer,
              dictionary_size: int,
              word_embeddings_dimension: int,
              max_word_length: int,
              word_embeddings: np.array):
        self.char_layer = char_layer
        chars = Input(shape=(max_word_length, ), name='chars')

        output = Dense(dictionary_size, weights=[word_embeddings], use_bias=False,
                       trainable=False, activation='softmax')
        output = output(Dense(word_embeddings_dimension, name='char_embed_to_word_embed')(self.char_layer(chars)))

        self.model = Model(inputs=chars, outputs=output)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
        print(self.model.summary())

    def train(self,
              dictionary: WordDictionary,
              char_set: str,
              test_part: float,
              random_seed: int,
              batch_size: int,
              max_word_len: int) -> None:
        """
        Обучение модели.

        :param dictionary: список слов.
        :param char_set: набор символов, для которых строятся эмбеддинги.
        :param test_part: на какой части выборки оценивать качество.
        :param random_seed: зерно для случайного генератора.
        :param batch_size: размер батча.
        :param max_word_len: максимальный учитываемый размер слова.
        """
        np.random.seed(random_seed)
        chars, y = self.prepare_words(dictionary, char_set, max_word_len)
        callbacks = [EarlyStopping(patience=3)]

        train_idx, test_idx = self.split_data_set(chars.shape[0], test_part)
        chars_train = chars[train_idx]
        y_train = y[train_idx]
        chars_val = chars[test_idx]
        y_val = y[test_idx]

        self.model.fit(chars_train, y_train, batch_size=batch_size, epochs=100, verbose=2,
                       validation_data=[chars_val, y_val], callbacks=callbacks)

    @staticmethod
    def split_data_set(sample_counter: int, test_part: float) -> Tuple[np.array, np.array]:
        perm = np.random.permutation(sample_counter)
        border = int(sample_counter * (1 - test_part))
        train_idx = perm[:border]
        test_idx = perm[border:]
        return train_idx, test_idx

    