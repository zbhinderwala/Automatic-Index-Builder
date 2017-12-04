#############################################
#   Imports
#############################################
import csv
import sys,glob,re
import pandas as pd

def generate_df(indexes,filenum):
    cols = ['word']
    index_data = pd.DataFrame(columns=cols)
    index_data['word']=indexes
    index_data['source']=filenum
    return index_data

df = pd.DataFrame()
if len(sys.argv) > 1:
    # Open given LaTeX file
    dirpath = sys.argv[1]+'/*'
    texfiles = glob.glob(dirpath)

    output_name = sys.argv[2]

    #print texfiles
    for t in range(0,len(texfiles)):
        raw_idx = []
        newcommandindex = ""
        T = open(texfiles[t],'r')
        for line in T:
            newCommand = re.search('\\\\newcommand{\\\\([A-Za-z]*)}(\[.+?\])?{(.*)}', line)
            if newCommand:
                if newCommand.group(3) and newCommand.group(3).__contains__('\index'):
                    newcommandindex = newCommand.group(1)

        print(newcommandindex)
        newindexlength = len(newcommandindex)
        newindexoffset = newindexlength + 2
        indextag = "\\" + newcommandindex + "{"

        T=open(texfiles[t],'r')
        tf = T.read()

        i = 0
        while i > -1:  # find contents of all the index tags
            i = tf.find("\index{", i) + 7
            if i != 6:
                j = tf.find("}", i)
                w = tf[i:j]
                if w not in raw_idx:
                    raw_idx.append(w)

            else:
                i = -1

        i = 0
        while i > -1:  # find contents of all the index tags with new command

            i = tf.find(indextag, i) + newindexoffset
            if i != newindexoffset-1:
                j = tf.find("}", i)
                w = tf[i:j]
                if w not in raw_idx:
                    raw_idx.append(w)

            else:
                i = -1

        df_temp = generate_df(raw_idx, t + 1)

        df = df.append(df_temp, ignore_index=True)

"""
    # use a nested dictionary to keep track of every term used
    final_idx = {}
    for entry in raw_idx:
        i = entry.find("!")
        j = entry.find("!", i + 1)

        if i == -1:
            i = len(entry)
        if entry[:i] not in final_idx:
            final_idx[entry[:i]] = {}

        if j == -1:
            j = len(entry)
        if i <> len(entry):
            if entry[i + 1:j] not in final_idx[entry[:i]]:
                final_idx[entry[:i]][entry[i + 1:j]] = {}

        if j <> len(entry):
            if entry[j + 1:] not in final_idx[entry[:i]][entry[i + 1:j]]:
                final_idx[entry[:i]][entry[i + 1:j]][entry[j + 1:]] = {}

    # Now sort and print
    level1 = final_idx.keys()
    level1.sort()

    for i in level1:
        #print i
        level2 = final_idx[i].keys()
        level2.sort()
        for j in level2:
            print "->", j
            level3 = final_idx[i][j].keys()
            level3.sort()
            for k in level3:
                k=k
"""




print('Exporting CSV...\n')
df.to_csv(output_name, index=False)
print('Index Data Generated....')