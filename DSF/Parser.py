#############################################
#   Imports
#############################################
import csv
import sys
import re
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text
import nltk
from TexSoup import TexSoup
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

    if word in nltk.corpus.stopwords.words('english'):
        return False

    pattern = re.compile("([A-Za-z0-9]+)")
    return pattern.match(word)


# split text using an array of delimiters and check validity
def splitText(text):
    # delimiters - only for characters that separate words
    # conjoining characters not included
    splitChars = {'\.', ' ', '\,', ':', ';', '\n', '\*' \
        '\|', '\?'}

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
if len(sys.argv) > 1:
    # Open given LaTeX file
    file = open(sys.argv[1], 'r')

    output_name = sys.argv[2]

else:
    # Open default LaTeX file
    file = open("chikin.tex", 'r')

    output_name = "CSV/text.csv"

# Get only the text from the file
text = LatexNodes2Text().latex_to_text(file.read().encode('utf-8'))

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

dataset2 = generateDataSet(bigrams)

df_final = df_final.append(dataset2, ignore_index = True)

##############################################
#   POS Tagging
##############################################


##############################################
#   Export CSV
##############################################
print('Exporting CSV...\n')
df_final.to_csv(output_name, index=False)

print('Dataset Generation Complete.')