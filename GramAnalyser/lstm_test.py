# -*- coding: utf-8 -*-
# Тестирование анализатора на различных выборках

import os
from typing import Dict

from GramAnalyser.analyser import Analyser
from GramAnalyser.settings import TEST_TAGGED_JZ, TEST_TAGGED_LENTA, TEST_TAGGED_VK, TEST_UNTAGGED_JZ, \
    TEST_UNTAGGED_LENTA, TEST_UNTAGGED_VK, TEST_GOLD_JZ, TEST_GOLD_LENTA, TEST_GOLD_VK, TEST_TAGGED_FOLDER
from GramAnalyser.test.evaluate_test_results import measure_test_accuracy

def test(analyser: Analyser, untagged_filename: str, tagged_filename: str):
    sentences = []
    with open(untagged_filename, "r", encoding='utf-8') as r:
        words = []
        for line in r:
            if line != "\n":
                records = line.strip().split("\t")
                word = records[1]
                words.append(word)
            else:
                sentences.append([word for word in words])
                words = []
    with open(tagged_filename, "w",  encoding='utf-8') as w:
        all_forms = analyser.analyse_sentences(sentences)
        for forms in all_forms:
            for i, form in enumerate(forms):
                line = "{}\t{}\t{}\t{}\t{}\n".format(str(i + 1), form.word, form.normal_form, form.pos, form.tag)
                w.write(line)
            w.write("\n")

def test_on_all_data_sets(analyser: Analyser) -> Dict:
    if not os.path.exists(TEST_TAGGED_FOLDER):
        os.makedirs(TEST_TAGGED_FOLDER)
    test(analyser, TEST_UNTAGGED_LENTA, TEST_TAGGED_LENTA)
    test(analyser, TEST_UNTAGGED_VK, TEST_TAGGED_VK)
    test(analyser, TEST_UNTAGGED_JZ, TEST_TAGGED_JZ)

    quality = dict()
    print("Lenta:")
    quality['Lenta'] = measure_test_accuracy(TEST_GOLD_LENTA, TEST_TAGGED_LENTA, True, None)
    print()
    print("VK:")
    quality['VK'] = measure_test_accuracy(TEST_GOLD_VK, TEST_TAGGED_VK, True, None)
    print()
    print("JZ:")
    quality['JZ'] = measure_test_accuracy(TEST_GOLD_JZ, TEST_TAGGED_JZ, True, None)
    print()
    print("All:")
    count_correct_tags = quality['Lenta'].correct_tags + quality['VK'].correct_tags + quality['JZ'].correct_tags
    count_correct_pos = quality['Lenta'].correct_pos + quality['VK'].correct_pos + quality['JZ'].correct_pos
    count_tags = quality['Lenta'].total_tags + quality['VK'].total_tags + quality['JZ'].total_tags
    count_correct_sentences = quality['Lenta'].correct_sentences + quality['VK'].correct_sentences + \
                              quality['JZ'].correct_sentences
    count_sentences = quality['Lenta'].total_sentences + quality['VK'].total_sentences + \
                      quality['JZ'].total_sentences
    quality['All'] = dict()
    quality['All']['tag_accuracy'] = float(count_correct_tags) / count_tags * 100
    quality['All']['pos_accuracy'] = float(count_correct_pos) / count_tags * 100
    quality['All']['sentence_accuracy'] = float(count_correct_sentences) / count_sentences * 100
    print("Точность грамм.значений {:.2f}%, точность частей речи {:.2f}%, точность разбора предложений {:.2f}%".format(quality['All']['tag_accuracy'], quality['All']['pos_accuracy'], quality['All']['sentence_accuracy']))
    return quality
