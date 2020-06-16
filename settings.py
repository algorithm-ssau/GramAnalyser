# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from pkg_resources import resource_filename

MODELS_FOLDER = resource_filename(__name__, "model_files")

FILES = dict()
FILES["build_config"] = "build_config.json"
FILES["train_config"] = "train_config.json"
FILES["train_model_config"] = "train_model.yaml"
FILES["train_model_weights"] = "train_model.h5"
FILES["eval_model_config"] = "eval_model.yaml"
FILES["eval_model_weights"] = "eval_model.h5"
FILES["gram_input"] = "gram_input.json"
FILES["gram_output"] = "gram_output.json"
FILES["word_dictionary"] = "word_dictionary.pickle"
FILES["char_set"] = "char_set.txt"
FILES["char_model_config"] = "char_model.yaml"
FILES["char_model_weights"] = "char_model.h5"

MODELS_PATHS = defaultdict(dict)

for key, file_name in FILES.items():
    MODELS_PATHS[key] = os.path.join(MODELS_FOLDER, file_name)