# -*- coding: utf-8 -*-
# Формирует обучающие наборы данных

from typing import List, Tuple
from collections import namedtuple

import nltk
import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from GramAnalyser.data_preparation.word_dictionary import WordDictionary
from GramAnalyser.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from GramAnalyser.config import TrainConfig, BuildModelConfig

WordForm = namedtuple("WordForm", "text gram_vector_index")

class TrainingSetGenerator:
    def __init__(self,
                 file_names: List[str],
                 config: TrainConfig,
                 grammeme_vectorizer_input: GrammemeVectorizer,
                 grammeme_vectorizer_output: GrammemeVectorizer,
                 indices: np.array,
                 word_dictionary: WordDictionary,
                 char_set: str,
                 build_config: BuildModelConfig):
        self.file_names = file_names  # type: List[str]
        # Параметры наборов.
        self.training_set_size = config.external_batch_size  # type: int
        self.bucket_borders = config.sentence_len_groups  # type: List[Tuple[int]]
        self.buckets = [list() for _ in range(len(self.bucket_borders))]
        self.build_config = build_config
        self.word_dictionary = word_dictionary
        self.char_set = char_set
        # Разбиение на выборки.
        self.indices = indices  # type: np.array
        # Подготовленные словари.
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.morph = MorphAnalyzer()  # type: MorphAnalyzer
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def __iter__(self):
        """
        Получение очередной выборки
        """
        last_sentence = []
        i = 0
        for filename in self.file_names:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        if i not in self.indices:
                            last_sentence = []
                            i += 1
                            continue
                        for index, bucket in enumerate(self.buckets):
                            if self.bucket_borders[index][0] <= len(last_sentence) < self.bucket_borders[index][1]:
                                bucket.append(last_sentence)
                            if len(bucket) >= self.training_set_size:
                                yield self.sentencesToFeaturesAndAnswers(bucket)
                                self.buckets[index] = []
                        last_sentence = []
                        i += 1
                    else:
                        _, word, _, pos, tags = line.split('\t')[0:5]
                        gram_vector_index = self.grammeme_vectorizer_output.get_index_by_name(pos + "#" + tags)
                        last_sentence.append(WordForm(text=word, gram_vector_index=gram_vector_index))
        for index, bucket in enumerate(self.buckets):
            yield self.sentencesToFeaturesAndAnswers(bucket)