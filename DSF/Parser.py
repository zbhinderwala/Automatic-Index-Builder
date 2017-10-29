import csv
import re
#from latex import LatesParser
#from tex2py import tex2py
from TexSoup import TexSoup
from pylatexenc.latex2text import LatexNodes2Text
#from plasTeX.TeX import TeX
from TexSoup import TexSoup
from plasTeX.Renderers.Text import Renderer
import plasTeX.

file=open('chikin.tex','r')
#soup=TexSoup(open('chikin.tex'))
#print (soup)
#Renderer().render(TeX(file).parse())
text = LatexNodes2Text().latex_to_text(file.read())
re.findall(text,r" ")
stripped_text = text.replace("\n"," ")
stripped_text.rstrip(" ")
print (stripped_text)
lines=stripped_text.split(" ")
#lines = (line.split(" ") for line in stripped_text if line)
for l in stripped_text:
    lines.append(l)
print (lines)
out_file=open('text.csv' , 'w')
writer = csv.writer(out_file)
writer.writerow(('title','intro'))
for l in stripped_text:
    writer.writerow(l)

