#############################################
#   Imports
#############################################

import csv
import sys
import re
import pandas as pd
from TexSoup import TexSoup
from pylatexenc.latex2text import LatexNodes2Text
from plasTeX.Renderers.Text import Renderer


#############################################
#   Helper methods
#############################################

# filter words that contain at least one letter/number
# useful for removing strings without letters
# "'" is not valid but "don't" is valid
def checkWords(word):
    pattern = re.compile("([A-Za-z0-9]+)")
    return pattern.match(word)


# split text using an array of delimiters and check validity
def splitText(text):
    # delimiters - only for characters that separate words
    # conjoining characters not included
    splitChars = {'\.', ' ', '\,', ':', ';', '\n', '\*'}

    splitString = ""
    for char in splitChars:
        splitString += char + '|'

    # spilt string
    words = re.split(splitString, text)

    # remove invalid words
    words = filter(checkWords, words)

    return words


# Create a dataset and preprocess its content
def generateDataSet(words):
    cols = ['words']
    data_set = pd.DataFrame(columns=cols)
    data_set['words'] = words

    # Get frequency of each word
    data_set['frequency'] = data_set.groupby('words')['words'].transform(pd.Series.value_counts)
    data_set = data_set.drop_duplicates()

    return data_set


#############################################
#   Main code
#############################################

if len(sys.argv) >= 0:
    # Open given LaTeX file
    file = open(sys.argv[1], 'r')

else:
    # Open default LaTeX file
    file = open("chikin.tex", 'r')

# Get only the text from the file
text = LatexNodes2Text().latex_to_text(file.read())

# Create list where each word is a separate element
words = splitText(text)

# create pandas DataFrame of words
data_set = generateDataSet(words)

data_set.to_csv("text.csv", index=False)
