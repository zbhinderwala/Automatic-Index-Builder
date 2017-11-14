#############################################
#   Imports
#############################################
import csv
import sys,glob,re
import pandas as pd

raw_idx=[]
if len(sys.argv) > 1:
    # Open given LaTeX file
    dirpath = sys.argv[1]+'/*'
    texfiles = glob.glob(dirpath)

    output_name = sys.argv[2]

    newcommandDict = {}
    """
    for line in file:
        newCommand = re.search('\\\\newcommand{\\\\([A-Za-z]*)}(\[.+?\])?{(.*)}', line)
        if newCommand :
            if newCommand.group(3) and newCommand.group(3).__contains__('\index'):
                    newcommandindex = newCommand.group(1)
    """



    #print texfiles
    for t in texfiles:
        T = open(t,'r')
        for line in T:
            newCommand = re.search('\\\\newcommand{\\\\([A-Za-z]*)}(\[.+?\])?{(.*)}', line)
            if newCommand:
                if newCommand.group(3) and newCommand.group(3).__contains__('\index'):
                    newcommandindex = newCommand.group(1)

        print(newcommandindex)
        newindexlength = len(newcommandindex)
        newindexoffset = newindexlength + 2
        indextag = "\\" + newcommandindex + "{"

        T=open(t,'r')
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


    print raw_idx

cols = ['word']
index_data = pd.DataFrame(columns=cols)
index_data['word']=raw_idx
print('Exporting CSV...\n')
index_data.to_csv(output_name, index=False)
print('Index Data Generated....')