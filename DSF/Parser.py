import csv
import re
import pandas as pd
#from latex import LatesParser
#from tex2py import tex2py
from TexSoup import TexSoup
from pylatexenc.latex2text import LatexNodes2Text
#from plasTeX.TeX import TeX
from TexSoup import TexSoup
from plasTeX.Renderers.Text import Renderer


file=open('chikin.tex','r')
#soup=TexSoup(open('chikin.tex'))
#print (soup)
#Renderer().render(TeX(file).parse())
text = LatexNodes2Text().latex_to_text(file.read())
#print text
re.findall(text,r" ")
stripped_text = text.replace("\n"," ")
#print stripped_text
stripped_text.rstrip(" ")
#print (stripped_text)
lines=stripped_text.split(" ")
lines=filter(None,lines)
print lines
#lines = (line.split(" ") for line in stripped_text if line)
#print (lines)
cols = ['words']
data_set=pd.DataFrame(columns = cols)
data_set['words'] = lines
data_set.to_csv("text.csv",index=False)
print data_set
#out_file=open('text.csv' , 'w')
#writer = csv.writer(out_file)
#writer.writerow(('title','intro'))
#for l in lines:
 #   writer.writerow(l)


