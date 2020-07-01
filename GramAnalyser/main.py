# -*- coding: utf-8 -*-
# Интерфейс для консольного приложения

import os, sys
from itertools import groupby
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.split(dir_path)[0])


### --- АНАЛИЗ ТЕКСТА --- ###
from GramAnalyser.analyser import Analyser
analyser = Analyser()
print("Введите текст:")
text = input()
words = analyser.split_text_on_words(text)
wordForms = analyser.analyse(words)
uniqieWordsDictionary = analyser.get_word_dictionary_for_text(wordForms)

uniqieWordsDictionary = sorted(uniqieWordsDictionary, key = lambda item: item.pos)
groupedWordDict = [list(i) for j, i in groupby(uniqieWordsDictionary, key = lambda item: item.pos)]

for group in groupedWordDict:
    print("")
    group = sorted(group, key = lambda item: item.frequency, reverse=True)
    for wordForm in group:
        print("{}       |     {}     |        {}".format(wordForm.normal_form, wordForm.pos, wordForm.frequency))
### ---           --- ###


"""
### -- ОБУЧЕНИЕ LSTM -- ###
from GramAnalyser.train import train
train()
### ---           --- ###
"""


"""
### --- ТЕСТ LSTM --- ###
from GramAnalyser.lstm_test import test_on_all_data_sets
from GramAnalyser.analyser import Analyser
analyser = Analyser()  
test_on_all_data_sets(analyser)
### ---           --- ###
"""