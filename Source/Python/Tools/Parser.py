#############################################
#   Imports
#############################################
import csv
import GoogleNgrams
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
import math
import argparse
import glob
import os
import phrasefinder
from GoogleNgrams import GoogleNgrams

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

############################################
# common words in english
############################################
common500 = ["the", "of", "and", "to", "a", "in", "that", "was", "he",
"i", "it", "his", "is", "with", "as", "for", "had", "you", "be", "on",
"not", "at", "but", "by", "her", "which", "this", "have", "from",
"she", "they", "all", "him", "were", "or", "are", "my", "we", "one",
"so", "their", "an", "me", "there", "no", "said", "when", "who",
"them", "been", "would", "if", "will", "what", "out", "more", "up",
"then", "into", "has", "some", "do", "could", "now", "very", "time",
"man", "its", "your", "our", "than", "about", "upon", "other", "only",
"any", "little", "like", "these", "two", "may", "did", "after", "see",
"made", "great", "before", "can", "such", "should", "over", "us",
"first", "well", "must", "mr", "down", "much", "good", "know", "where",
"old", "men", "how", "come", "most", "never", "those", "here", "day",
"came", "way", "own", "go", "life", "long", "through", "many", "being",
"himself", "even", "shall", "back", "make", "again", "every", "say",
"too", "might", "without", "while", "same", "am", "new", "think",
"just", "under", "still", "last", "take", "went", "people", "away",
"found", "yet", "thought", "place", "hand", "though", "small", "eyes",
"also", "house", "years", "-", "another", "don't", "young", "three",
"once", "off", "work", "right", "get", "nothing", "against", "left",
"ever", "part", "let", "each", "give", "head", "face", "god", "0",
"between", "world", "few", "put", "saw", "things", "took", "letter",
"tell", "because", "far", "always", "night", "mrs", "love", "both",
"sir", "why", "look", "having", "mind", "father", "called", "side",
"looked", "home", "find", "going", "whole", "seemed", "however",
"country", "got", "thing", "name", "among", "seen", "heart", "told",
"done", "king", "water", "asked", "heard", "soon", "whom", "better",
"something", "knew", "lord", "course", "end", "days", "moment",
"enough", "almost", "general", "quite", "until", "thus", "hands",
"nor", "light", "room", "since", "woman", "words", "gave", "b",
"mother", "set", "white", "taken", "given", "large", "best", "brought",
"does", "next", "whose", "state", "yes", "oh", "door", "turned",
"others", "poor", "power", "present", "want", "perhaps", "death",
"morning", "la", "rather", "word", "miss", "less", "during", "began",
"themselves", "felt", "half", "lady", "full", "voice", "cannot",
"feet", "order", "near", "true", "1", "it's", "matter", "stood",
"together", "year", "used", "war", "till", "use", "thou", "son",
"high", "round", "above", "certain", "often", "kind", "indeed", "i'm",
"along", "case", "fact", "myself", "children", "anything", "four",
"dear", "keep", "nature", "known", "point", "p", "friend", "says",
"passed", "within", "land", "sent", "church", "believe", "girl",
"city", "times", "form", "herself", "therefore", "hundred", "john",
"wife", "fire", "several", "body", "sure", "money", "means", "air",
"open", "held", "second", "gone", "already", "least", "alone", "hope",
"thy", "chapter", "whether", "boy", "english", "itself", "2", "women",
"hear", "cried", "leave", "either", "number", "rest", "child",
"behind", "read", "lay", "black", "government", "friends", "became",
"around", "river", "sea", "ground", "help", "c", "i'll", "short",
"question", "reason", "become", "call", "replied", "town", "family",
"england", "lost", "speak", "answered", "five", "coming", "possible",
"making", "hour", "dead", "really", "looking", "law", "captain",
"different", "manner", "business", "states", "earth", "st", "human",
"early", "sometimes", "spirit", "care", "sat", "public", "close",
"towards", "kept", "french", "party", "truth", "line", "strong",
"book", "able", "later", "return", "hard", "mean", "feel", "story",
"m", "received", "following", "fell", "wish", "person", "beautiful",
"seems", "dark", "history", "followed", "subject", "thousand", "ten",
"returned", "thee", "age", "turn", "fine", "across", "show", "arms",
"character", "live", "soul", "met", "evening", "die", "common",
"ready", "suddenly", "doubt", "bring", "ii", "red", "that's",
"account", "cause", "necessary", "can't", "need", "answer", "miles",
"carried", "although", "fear", "hold", "interest", "force",
"illustration", "sight", "act", "master", "ask", "idea", "ye", "sense",
"an'", "art", "position", "rose", "3", "company", "road", "further",
"nearly", "table"]

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

    if word.lower() in common500:
        return False

    if word.isdigit():
        return False

    if len(word) < 2:
        return False

    if any(i.isdigit() for i in word):
        return False
    
    invalidChars = {'=', '<', '>', '+'}

    for c in invalidChars:
        if c in word:
            return False

    if len(word) < 3:
        return False

    pattern = re.compile("([A-Za-z]+)")
    return pattern.match(word)

def checkBigram(bigram):
    #removing invalid words
    if not checkWords(bigram[0]) or not checkWords(bigram[1]):
        return False
    else:
        return True


# split text using an array of delimiters and check validity
def splitText(text):
    # delimiters - only for characters that separate words
    # conjoining characters not included
    splitChars = {'\.', ' ', '\,', ':', ';', '\n', '\*', \
        '\|', '\?', '\(', '\)', '\^', '_','!','~','$','@','^',\
        '{','}','\[','\]'
    }

    splitString = ""
    for char in splitChars:
        splitString += char + '|'

    # spilt string
    words = re.split(splitString, text)

    return words


# Create a dataset and preprocess its content
def generateDataSet(words, type='unigram'):
    cols = ['word']
    data_set = pd.DataFrame(columns=cols)

    if (type is 'unigram'):
        data_set['word'] = words
        data_set['n_gram_score'] = 2

    if (type is 'bigram'):
        newWords = []
        for bigram in words:
            newWords.append(' '.join(bigram))
        data_set['word'] = newWords
        data_set['n_gram_score'] = 10

    return data_set

def generate_csv(file, filenum=1):
    print('Reading LaTeX File: ' + os.path.basename(file.name) + '...\n')

    # Get only the text from the file
    try:
        latextext= file.read()
    except:
        latextext= file.read().encode("utf-8")

    text = LatexNodes2Text().latex_to_text(latextext)
    #lw = LatexWalker(latextext)

    ##############################################
    #   UNIGRAMS
    ##############################################
    print('Gathering Unigrams...\n')
    # Create list where each word is a separate element
    words = splitText(text)
    # remove invalid words
    unigrams = filter(checkWords, words)

    # create pandas DataFrame of words
    data_set = generateDataSet(unigrams)
    df_final = data_set

    #############################################
    #   BIGRAMS
    #############################################
    print('Gathering Bigrams...\n')
    bigrams = list(nltk.bigrams(words))

    bigrams = filter(checkBigram, bigrams)

    dataset2 = generateDataSet(bigrams, type='bigram')

    df_final = df_final.append(dataset2, ignore_index = True)

    ##############################################
    #   POS Tagging
    ##############################################
    print('Determining Parts of Speech...\n')
    pos_tags = list(nltk.pos_tag(df_final['word']))
    pos_df = pd.DataFrame(pos_tags, columns=['word', 'pos'])

    df_final = pd.concat([df_final, pos_df['pos']], axis=1)
    df_final['pos'] = df_final['pos'].replace(np.nan, 'NA')

    ##############################################
    #   Frequencies
    ##############################################
    print('Calculate Frequencies...\n')

    # Get frequency of each word
    df_final = df_final.groupby(['word','n_gram_score','pos']).size().reset_index(name="frequency")
    df_final['wordCount'] = len(words)

    ##############################################
    #   Calculating Tf and Idf
    ##############################################
    """
    print('Calculate Term Frequency and Document Frequency...')
    
    word_count = len(words)
    term_freq = []
    term_freq = df_final['frequency'] / word_count
    idf = 17 / 1
    tf_idf = []
    tf_idf = term_freq * idf
    df_final['tf_idf'] = tf_idf
    """
    ##############################################
    #   Calculating Informativeness
    ##############################################
    """
    print('Calculate Informativeness...')
    
    inf_list = []
    for i in term_freq:
        inf = i * math.log(i * idf , 2)
        inf_list.append(inf)
    df_final['inf'] = inf_list
    """
    ##############################################
    #   Google Ngram - Match Count and Volume Count
    ##############################################

    if args.ngram:
        print('Retrieving Google Ngram Data...')

        match = []
        volume = []

        match, volume = GoogleNgrams(df_final['word'], quiet=False)

        df_final['match_count'] = match
        df_final['volume_count'] = volume

        print('\n')


    ##############################################
    #   Removing the words based on parts of speech
    ##############################################
    """pos_list = ['CD','CC','IN','PRP','RB','VBD','VB','JJ','VBN','VBZ','VBG','IN','WDT','WRB','VBP','FW']
    for i in df_final['pos']:
        if i in pos_list:
            df_final = df_final.drop()"""

    ##############################################
    #   Check Formatting
    ##############################################

    ##############################################
    #   File source
    ##############################################

    df_final['source'] = filenum

    ##############################################
    #   Export CSV
    ##############################################

    # Sort by frequency
    df_final = df_final.sort_values(['frequency', 'word'], ascending = [False, True])

    return df_final

#############################################
#   Main code
#############################################
parser = argparse.ArgumentParser(prog='Parser.py', description='Parse a LaTeX file into a CSV file to determine file index.')
parser.add_argument('-d', '--dir', help='Parse entire directory of LaTeX files', required=False)
parser.add_argument('-f', '--file', help='Parse single LaTeX file', required=False)
parser.add_argument('-o', '--output', help='Output directory')
parser.add_argument('-n', '--ngram', help='Calculate Google Ngram data', required=False, action='store_true', default=False)

args = parser.parse_args()

df = pd.DataFrame()

if args.file:
    # Open given LaTeX file
    file = open(args.file, 'r')

    df = generate_csv(file)

elif args.dir:
    # Open one LaTeX file at a time
    dirpath = args.dir + "/*.tex"
    texfiles = glob.glob(dirpath)

    for i in range(0, len(texfiles)):
        file = open(texfiles[i], 'r')
        
        df_temp = generate_csv(file, i + 1)

        df = df.append(df_temp, ignore_index=True)

else:
    parser.print_help()
    exit()

print('Exporting CSV...\n')
df.to_csv(args.output, index=False)

print('Dataset Generation Complete.')