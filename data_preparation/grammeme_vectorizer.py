# -*- coding: utf-8 -*-
# Векторизатор граммем. Собирает возможные грамматические значения по корпусу и на их основе строит грамматические вектора.

import jsonpickle
from collections import defaultdict
from typing import Dict, List, Set

from GramAnalyser.data_preparation.process_tag import filter_gram_tag

def get_empty_category():
    return {GrammemeVectorizer.UNKNOWN_VALUE}

class GrammemeVectorizer(object):
    UNKNOWN_VALUE = "Unknown"

    def __init__(self):
        self.all_grammemes = defaultdict(get_empty_category)  # type: Dict[str, Set]
        self.vectors = []  # type: List[List[int]]
        self.name_to_index = {}  # type: Dict[str, int]

    def collect_grammemes(self, filename: str) -> None:
        """
        Собрать возможные грамматические значения по файлу с морфоразметкой.
        
        :param filename: файл с морфоразметкой.
        """
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                pos_tag, grammemes = line.split("\t")[3:5]
                self.add_grammemes(pos_tag, grammemes)

    def add_grammemes(self, pos_tag: str, gram: str) -> int:
        """
        Добавить новое грамматическое значение в список известных
        """
        gram = filter_gram_tag(gram)
        vector_name = pos_tag + '#' + gram
        if vector_name not in self.name_to_index:
            self.name_to_index[vector_name] = len(self.name_to_index)
            self.all_grammemes["POS"].add(pos_tag)
            gram = gram.split("|") if gram != "_" else []
            for grammeme in gram:
                category = grammeme.split("=")[0]
                value = grammeme.split("=")[1]
                self.all_grammemes[category].add(value)
        return self.name_to_index[vector_name]

    def init_possible_vectors(self) -> None:
        """
        Инициализировать все возможные векторы по известным грамматическим значениям
        """
        self.vectors = []
        for grammar_val, index in sorted(self.name_to_index.items(), key=lambda x: x[1]):
            pos_tag, grammemes = grammar_val.split('#')
            grammemes = grammemes.split("|") if grammemes != "_" else []
            vector = self.__build_vector(pos_tag, grammemes)
            self.vectors.append(vector)

    def get_vector(self, vector_name: str) -> List[int]:
        """
        Получить вектор по грамматическим значениям.
        
        :param vector_name: часть речи + грамматическое значение.
        :return: вектор.
        """
        if vector_name not in self.name_to_index:
            return [0] * len(self.vectors[0])
        return self.vectors[self.name_to_index[vector_name]]

    def get_vector_by_index(self, index: int) -> List[int]:
        """
        Получить вектор по индексу
        
        :param index: индекс.
        :return: вектор.
        """
        return self.vectors[index] if 0 <= index < len(self.vectors) else [0] * len(self.vectors[0])

    def get_ordered_grammemes(self) -> List[str]:
        """
        Получить упорядоченный список возможных грамматических значений.
        
        :return: список грамматических значений.
        """
        flat = []
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            for value in sorted(list(values)):
                flat.append(category+"="+value)
        return flat

    def size(self) -> int:
        return len(self.vectors)

    def grammemes_count(self) -> int:
        return len(self.get_ordered_grammemes())

    def is_empty(self) -> int:
        return len(self.vectors) == 0

    def get_name_by_index(self, index):
        d = {index: name for name, index in self.name_to_index.items()}
        return d[index]

    def get_index_by_name(self, name):
        pos = name.split("#")[0]
        gram = filter_gram_tag(name.split("#")[1])
        return self.name_to_index[pos + "#" + gram]

    def __build_vector(self, pos_tag: str, grammemes: List[str]) -> List[int]:
        """
        Построение вектора по части речи и грамматическим значениям.
        
        :param pos_tag: часть речи.
        :param grammemes: грамматические значения.
        :return: вектор.
        """
        vector = []
        gram_tags = {pair.split("=")[0]: pair.split("=")[1] for pair in grammemes}
        gram_tags["POS"] = pos_tag
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            if category not in gram_tags:
                vector += [1 if value == GrammemeVectorizer.UNKNOWN_VALUE else 0 for value in sorted(list(values))]
            else:
                vector += [1 if value == gram_tags[category] else 0 for value in sorted(list(values))]
        return vector

    def save(self, dump_filename: str) -> None:
        with open(dump_filename, "w") as f:
            f.write(jsonpickle.encode(self, f))

    def load(self, dump_filename: str):
        with open(dump_filename, "r") as f:
            vectorizer = jsonpickle.decode(f.read())
            self.__dict__.update(vectorizer.__dict__)
