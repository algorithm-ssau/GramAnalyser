# -*- coding: utf-8 -*-

import numpy as np

from GramAnalyser.data_preparation.word_dictionary import WordDictionary

def load_embeddings(embeddings_file_name: str, dictionary: WordDictionary, word_count: int):
    ###
    ### Загрузка словных эмбэндингов из файла
    ###
    with open(embeddings_file_name, "r", encoding='utf-8') as f:
        line = next(f)
        dimension = int(line.strip().split()[1])
        matrix = np.random.rand(min(dictionary.size(), word_count+1), dimension) * 0.05
        words = {word: i for i, word in enumerate(dictionary.words[:word_count])}
        for line in f:
            try:
                word = line.strip().split()[0]
                embedding = [float(i) for i in line.strip().split()[1:]]
                index = words.get(word)
                if index is not None:
                    matrix[index] = embedding
            except ValueError or UnicodeDecodeError:
                continue
        return matrix