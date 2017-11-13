#############################################
#   Imports
#############################################
import csv
import sys
import re
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text
import numpy as np
from pylatexenc.latexwalker import (
    MacrosDef, LatexWalker, LatexToken, LatexCharsNode, LatexGroupNode, LatexCommentNode,
    LatexMacroNode, LatexEnvironmentNode, LatexMathNode, LatexWalkerParseError
)
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

#############################################
#   Helper methods
#############################################

# filter words that contain at least one letter/number
# useful for removing strings without letters
# "'" is not valid but "don't" is valid
def checkWords(word):
    try:
        word = str(word)
    except:
        return False

    if word.lower() in nltk.corpus.stopwords.words('english'):
        return False

    if word.isdigit():
        return False

    if len(word) < 2:
        return False

    pattern = re.compile("([A-Za-z0-9]+)")
    return pattern.match(word)


# split text using an array of delimiters and check validity
def splitText(text):
    # delimiters - only for characters that separate words
    # conjoining characters not included
    splitChars = {'\.', ' ', '\,', ':', ';', '\n', '\*' \
        '\|', '\?', '\(', '\)', '\^', '_'}

    splitString = ""
    for char in splitChars:
        splitString += char + '|'

    # spilt string
    words = re.split(splitString, text)

    # remove invalid words
    words = filter(checkWords, words)

    return words


# Create a dataset and preprocess its content
def generateDataSet(words, type='unigram'):
    cols = ['word']
    data_set = pd.DataFrame(columns=cols)

    if (type is 'unigram'):
        data_set['word'] = words

    if (type is 'bigram'):
        newWords = []
        for bigram in words:
            newWords.append(' '.join(bigram))
        data_set['word'] = newWords

    return data_set

#def getKeywords(fullText):



#############################################
#   Main code
#############################################
if len(sys.argv) > 1:
    # Open given LaTeX file
    file = open(sys.argv[1], 'r')

    output_name = sys.argv[2]

else:
    # Open default LaTeX file
    file = open("chikin.tex", 'r')

    output_name = "CSV/text.csv"

print('Reading LaTeX File...\n')

# Get only the text from the file
latextext = file.read().encode('utf-8')
text = LatexNodes2Text().latex_to_text(latextext)
lw = LatexWalker(latextext)

##############################################
#   UNIGRAMS
##############################################
print('Gathering Unigrams...\n')
# Create list where each word is a separate element
words = splitText(text)

# create pandas DataFrame of words
data_set = generateDataSet(words)
df_final = data_set

#############################################
#   BIGRAMS - Commented until improved efficiency
#############################################
print('Gathering Bigrams...\n')
bigrams = list(nltk.bigrams(words))

dataset2 = generateDataSet(bigrams, type='bigram')

df_final = df_final.append(dataset2, ignore_index = True)

##############################################
#   POS Tagging
##############################################
print('Determining Parts of Speech...\n')
pos_tags = list(nltk.pos_tag(words))
pos_df = pd.DataFrame(pos_tags, columns=['word', 'pos'])

df_final = pd.concat([df_final, pos_df['pos']], axis=1)
df_final['pos'] = df_final['pos'].replace(np.nan, 'NA')

##############################################
#   Frequencies
##############################################
print('Calculate Frequencies...\n')

# Get frequency of each word
df_final = df_final.groupby(['word', 'pos']).size().reset_index(name="frequency")

##############################################
#   Check Keywords
##############################################

print ('Checking Keywords...\n')

##############################################
#   Check Formatting
##############################################


##############################################
#   Export CSV
##############################################

# Sort by frequency
df_final = df_final.sort_values(['frequency', 'word'], ascending = [False, True])

print('Exporting CSV...\n')
df_final.to_csv(output_name, index=False)

print('Dataset Generation Complete.')