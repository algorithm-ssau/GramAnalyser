import sys
import getopt
from collections import defaultdict, namedtuple

def read_sents(infile):
    answer, curr_sent = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    answer.append(curr_sent)
                curr_sent = []
                continue
            splitted = line.split("\t")
            if len(splitted) == 5:
                word, lemma, pos, tags = splitted[1:]
            elif len(splitted) == 4:
                word, pos, tags = splitted[1:]
                lemma = None
            else:
                raise ValueError("Each line should have 4 or 5 columns")
            if tags != "_":
                tags = dict(elem.split("=") for elem in tags.split("|"))
            else:
                tags = dict()
            curr_sent.append([word, pos, tags, lemma])
    if len(curr_sent) > 0:
        answer.append(curr_sent)
    return answer

POS_TO_MEASURE = ["NOUN", "PRON", "DET", "ADJ", "VERB", "NUM"]
DOUBT_ADVERBS = ["как", "когда", "пока", "так", "где"]

def get_grams_for_pos(pos):
    if pos == "NOUN":
        return ["Gender", "Number", "Case"]
    elif pos == "ADJ":
        return ["Gender", "Number", "Case", "Variant", "Degree"]
    elif pos == "PRON":
        return ["Gender", "Number", "Case"]
    elif pos == "DET":
        return ["Gender", "Number", "Case"]
    elif pos == "VERB":
        return ["Gender", "Number", "VerbForm", "Mood", "Tense"]
    elif pos == "ADV":
        return ["Degree"]
    elif pos == "NUM":
        return ["Gender", "Case", "NumForm"]
    else:
        return []

VALUE_ALIASES = {'Brev': ["Short", "Brev"], 'Short': ["Short", "Brev"], "Notpast": ["Pres", "Fut"], "NumForm": ["Form"]}

def are_equal_tags(pos, first, second):
    grams_for_pos = get_grams_for_pos(pos)
    for pos, value in first.items():
        if pos in grams_for_pos:
            second_value = second.get(pos)
            if not (second_value == value or second_value in VALUE_ALIASES.get(value, [])):
                return False
    return True

def measure_accuracy(gold, test, measure_lemmas=False):
    correct_tags, correct_sents_by_tags, total_tags, correct_pos = 0, 0, 0, 0
    answer = dict()
    correct, correct_sents = 0, 0
    incorrect_matches = defaultdict(int)
    for i, corr_sent in enumerate(gold):
        is_correct_tags_sent = True
        is_correct_sent = True
        for word, corr_pos, corr_tags, corr_lemma in corr_sent:
            if corr_pos in POS_TO_MEASURE or (corr_pos == "ADV" and word.lower() not in DOUBT_ADVERBS):
                total_tags += 1
        if i >= len(test):
            continue
        for corr_elem, test_elem in zip(corr_sent, test[i]):
            word = corr_elem[0].lower()
            corr_pos, corr_tags, corr_lemma = corr_elem[1:]
            test_pos, test_tags, test_lemma = test_elem[1:]
            if corr_pos not in POS_TO_MEASURE:
                if corr_pos != "ADV" or word in DOUBT_ADVERBS:
                    continue
            if (corr_pos == test_pos) or (corr_pos == "NOUN" and test_pos == "PROPN"):
                correct_pos += 1
                tag_match = are_equal_tags(corr_pos, corr_tags, test_tags)
                if tag_match and measure_lemmas and test_lemma is not None:
                    lemma_match = (corr_lemma.replace("ё", "е").lower()
                                   == test_lemma.replace("ё", "е").lower())
                else:
                    lemma_match = False
            else:
                tag_match, lemma_match = False, False
            correct_tags += int(tag_match)
            if not tag_match:
                corr_tag_string = (corr_pos + "," +
                                   "|".join("=".join(elem) for elem in sorted(corr_tags.items())))
                test_tag_string = (test_pos + "," +
                                   "|".join("=".join(elem) for elem in sorted(test_tags.items())))
                incorrect_matches[(corr_tag_string, test_tag_string)] += 1
            correct += int(lemma_match)
            is_correct_tags_sent &= tag_match
            is_correct_sent &= lemma_match
        correct_sents_by_tags += int(is_correct_tags_sent)
        correct_sents += int(is_correct_sent)
    answer['correct_tags'] = correct_tags
    answer['correct_pos'] = correct_pos
    answer['correct_sents_by_tags'] = correct_sents_by_tags
    answer['total_tags'], answer['total_sents'] = total_tags, len(gold)
    answer['incorrect_matches'] = incorrect_matches
    if measure_lemmas:
        answer['correct'], answer['correct_sents'] = correct, correct_sents
    return answer

def measure_test_accuracy(gold_file, test_file, measure_lemmas, dump_file):
    gold, test = read_sents(gold_file), read_sents(test_file)
    quality = measure_accuracy(gold, test, measure_lemmas=measure_lemmas)
    correct_tags, total_tags, correct_sents_by_tags, correct_pos = \
        quality['correct_tags'], quality['total_tags'], quality['correct_sents_by_tags'], quality['correct_pos']
    total_sents = len(gold)
    tag_accuracy = 100 * (correct_tags / total_tags)
    pos_accuracy = 100 * (correct_pos / total_tags)
    sentence_accuracy = 100 * (correct_sents_by_tags / total_sents)
    print("{} грамм.значений из {}, точность {:.2f}%".format(correct_tags, total_tags, tag_accuracy))
    print("{} частей речи из {}, точность {:.2f}%".format(correct_pos, total_tags, pos_accuracy))
    print("{} предложений из {}, точность {:.2f}%".format(correct_sents_by_tags, total_sents, sentence_accuracy))
    full_accuracy = None
    full_sentence_accuracy = None
    if measure_lemmas:
        correct, correct_sents = quality['correct'], quality['correct_sents']
        full_accuracy = 100 * (correct / total_tags)
        full_sentence_accuracy = 100 * (correct_sents / total_sents)
        print("Качество полного разбора:")
        print("{} слов из {}, точность {:.2f}%".format(correct, total_tags, full_accuracy))
        print("{} предложений из {}, точность {:.2f}%".format(correct_sents, total_sents, full_sentence_accuracy))
    if dump_file is not None:
        with open(dump_file, "w", encoding="utf8") as fout:
            for (first, second), count in sorted(
                    quality['incorrect_matches'].items(), key=(lambda x: x[1]), reverse=True):
                fout.write("{}\t{}\n{}\n\n".format(first, count, second))
    Accuracy = namedtuple('Accuracy', 'tag_accuracy sentence_accuracy full_tag_accuracy full_sentence_accuracy '
                                      'correct_tags total_tags correct_sentences total_sentences correct_pos '
                                      'pos_accuracy')
    return Accuracy(tag_accuracy=tag_accuracy, sentence_accuracy=sentence_accuracy,
                    full_tag_accuracy=full_accuracy, full_sentence_accuracy=full_sentence_accuracy,
                    correct_tags=correct_tags, total_tags=total_tags,
                    correct_sentences=correct_sents_by_tags, total_sentences=total_sents,
                    correct_pos=correct_pos, pos_accuracy=pos_accuracy)