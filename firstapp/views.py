# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import HttpResponse
from .forms import UserForm
from GramAnalyser.analyser import Analyser
import os, sys
from itertools import groupby

def index(request):
    userform = UserForm()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.split(dir_path)[0])

    if request.method == "POST":
        text = request.POST.get("text")     
        analyser = Analyser()
        words = analyser.split_text_on_words(text)
        wordForms = analyser.analyse(words)
        uniqieWordsDictionary = analyser.get_word_dictionary_for_text(wordForms)

        uniqieWordsDictionary = sorted(uniqieWordsDictionary, key = lambda item: item.pos)
        groupedWordDict = [list(i) for j, i in groupby(uniqieWordsDictionary, key = lambda item: item.pos)]

        
        for group in groupedWordDict:
           group = sorted(group, key = lambda item: item.frequency, reverse=True)

        return render(request, "index.html", {"text": text, "groups": groupedWordDict, "form": userform})
    else:
        return render(request, "index.html", {"text": "", "form": userform})
