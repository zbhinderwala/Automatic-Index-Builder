import csv
import sys
import re
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text
import numpy as np
import distance
import phrasefinder
import subprocess
from pylatexenc.latexwalker import (
    MacrosDef, LatexWalker, LatexToken, LatexCharsNode, LatexGroupNode, LatexCommentNode,
    LatexMacroNode, LatexEnvironmentNode, LatexMathNode, LatexWalkerParseError
)
import nltk
import math
import argparse
import glob
import os
import sklearn.cluster

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
    # removing invalid words
    if not checkWords(bigram[0]) or not checkWords(bigram[1]):
        return False
    else:
        return True


# split text using an array of delimiters and check validity
def splitText(text):
    # delimiters - only for characters that separate words
    # conjoining characters not included
    splitChars = {'\.', ' ', '\,', ':', ';', '\n', '\*', \
                  '\|', '\?', '\(', '\)', '\^', '_', '!', '~', '$', '@', '^', \
                  '{', '}', '\[', '\]'
                  }

    splitString = ""
    for char in splitChars:
        splitString += char + '|'

    # spilt string
    words = re.split(splitString, text)

    return words

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

def clustering(words):
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    centroids = []
    clusters=[]
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        centroids.append(exemplar)
        clusters.append(cluster_str)
    cols = ['word', 'clusters']
    cluster_df = pd.DataFrame(columns=cols)
    cluster_df['word'] = centroids
    cluster_df['clusters'] = clusters
    cluster_df['source']=1
    return cluster_df

def ngrams(words,quiet=True):
    match = []
    volume = []

    word_count = len(words)
    counter = 1
    for x in words:

        if not quiet:
            sys.stdout.write("\r%d/%d" % (counter, word_count))
            sys.stdout.flush()

        match_str = '1'
        vol_str = '1'

        try:
            # search for term x through Google Ngrams using phrasefinder
            result = phrasefinder.search(x)

            if result.status == phrasefinder.Status.Ok:
                if len(result.phrases) > 0:
                    match_str = (result.phrases[0].match_count)
                    vol_str = (result.phrases[0].volume_count)
        except:
            match_str = '-1'
            vol_str = '-1'

        match.append(match_str)
        volume.append(vol_str)

        counter += 1

    return match, volume


def parsing(file, filenum=1):
    print('Reading LaTeX File: ' + os.path.basename(file.name) + '...\n')

    # Get only the text from the file
    try:
        latextext = file.read()
    except:
        latextext = file.read().encode("utf-8")

    text = LatexNodes2Text().latex_to_text(latextext)
    # lw = LatexWalker(latextext)

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

    df_final = df_final.append(dataset2, ignore_index=True)

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
    df_final = df_final.groupby(['word', 'n_gram_score', 'pos']).size().reset_index(name="frequency")
    df_final['wordCount'] = len(words)

    ##############################################
    # Clustering
    ##############################################
    print ('Clustering the words...\n')
    df_clusters = clustering(np.array(df_final['word']))

    ## merging the Dataframe and clusters
    df_final['word'] = df_final['word'].str.lower()
    df_final = df_final.drop_duplicates(['word'], keep='first')

    #### Merging the clusters and training to include only the centroids as a part of training set ####
    df_merged = pd.merge(df_final, df_clusters, how='left', left_on=['word'],
                              right_on=['word'])
    df_merged = df_merged.dropna()

    ##############################################
    # Calculating Google Ngrams value
    ##############################################

    print('Retrieving Google Ngram Data...\n')

    match = []
    volume = []

    match, volume = ngrams(df_merged['word'], quiet=False)

    df_merged['match_count'] = match
    df_merged['volume_count'] = volume

    ##############################################
    #   Calculating Tf and Idf
    ##############################################
    print('\n')
    print('Calculate Term Frequency and Document Frequency...')
    #### Processing the data - adding features ####

    #### Adding tf_idf ( term frequency - inverse document frequency ) to the training set ####
    doc_count = 3332956
    df_merged = df_merged.fillna(1)
    df_merged['volume_count'] = pd.to_numeric(df_merged.volume_count, errors='coerce')
    df_merged['tf_idf'] = df_merged['frequency'] / df_merged['wordCount']
    df_merged['tf_idf'] = df_merged['tf_idf'] * (df_merged['volume_count'].apply(lambda x: math.log(doc_count / (2 + x), 10)))

    #### Adding the pos_rank - rank for every parts of speech ####
    df_merged['pos_rank'] = 2
    df_merged.loc[df_merged.pos == 'NN', 'pos_rank'] = 14
    df_merged.loc[df_merged.pos == 'JJ', 'pos_rank'] = 7
    df_merged.loc[df_merged.pos == 'NNS', 'pos_rank'] = 6
    df_merged.loc[df_merged.pos == 'NNP', 'pos_rank'] = 6
    df_merged.loc[df_merged.pos == 'VBP', 'pos_rank'] = 3

    ###############################################
    # Scoring Function
    ##############################################

    #### Scoring the training properties ####
    df_merged['score'] = 0
    for index, row in df_merged.iterrows():
        score = ((row['pos_rank'] * row['n_gram_score'])) * row['tf_idf']
        df_merged.loc[index, 'score'] = score

    df_merged = df_merged.sort_values(['score'], ascending=False)
    ##############################################
    #   Calculating Probabilility using Logistic Regression Coefficients
    ##############################################

    df_merged['probability'] = 0
    for index, row in df_merged.iterrows():
        prob = 1 / (1 + math.exp(-(4.33686317 * row['score'])))
        df_merged.loc[index, 'probability'] = prob

    df_merged.sort_values(['probability'], ascending=False)


    return df_merged


def makeindex(file,index):
    with open(file, 'r') as tf_in:
        tf = tf_in.readlines()

    for i in range(len(tf)):
        bg = tf[i].find('\\begin{document}')
        if bg != -1:
            text = '\\usepackage{makeidx}' + '\n' + '\\makeindex' + '\n' + tf[i]
            tf[i] = tf[i].replace('\\begin{document}', text)
        end = tf[i].find('\\end{document}')
        if end != -1:
            printText = '\\printindex' + '\n' + tf[i]
            tf[i] = tf[i].replace('\\end{document}', printText)

    for w in index:
        for i in range(len(tf)):
            if w in tf[i]:
                tf[i] = tf[i].replace(w, w + ' \index{' + w + '}')

    with open(file, 'w') as tf_out:
        tf_out.writelines(tf)

    cmd1 = 'pdflatex %s' % (file)
    filename = file[:file.rfind('.')]+'.idx'
    cmd2 = 'makeindex %s' %(filename)
    subprocess.call(cmd1, shell=True)
    subprocess.call(cmd2, shell=True)
    subprocess.call(cmd1, shell=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file = open(sys.argv[1], 'r')
        index_count=50
        if sys.argv[2]:
            index_count=int(sys.argv[2])
        df=pd.DataFrame()
        if file:
            df=parsing(file)
            indexes = df['word'][:index_count]
            df['word'][:index_count].to_csv('Indexes.csv',index=True)
            makeindex(sys.argv[1],indexes.tolist())

    else:
        print('Please give a valid file name')

