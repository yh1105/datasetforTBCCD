from os import listdir
from os.path import isfile, join
import random
positiveFiles=[]
for i in range(0,15):
    positiveFiles.append([])
for i in range(0,15):
    s="./data/"+str(i+1)+"/"
    positiveFiles[i]=[s + f for f in listdir(s) if isfile(join(s, f))]
print(positiveFiles)
listrecTotFileName=[]
listtempT=[]
for i in range(0,15):
    for j in range(0,len(positiveFiles[i])):
        listtempT.append(positiveFiles[i][j])
        listtempT.append(str(i+1))
        listrecTotFileName.append(listtempT)
        listtempT=[]
random.shuffle(listrecTotFileName)
print(listrecTotFileName)
flistc=open("flistPOJ.txt",'w')
for i in range(0,len(listrecTotFileName)):
    flistc.write(listrecTotFileName[i][0])
    flistc.write("\t")
flistc.close()
ftrain=open("getdatatrainPOJ.txt",'w')
fdev=open("getdatadevPOJ.txt",'w')
ftest=open("getdatatestPOJ.txt",'w')
for i in range(0,500):
    ftest.write(listrecTotFileName[i][0])
    ftest.write('\t')
    ftest.write(listrecTotFileName[i][1])
    ftest.write('\n')
for i in range(500,1000):
    fdev.write(listrecTotFileName[i][0])
    fdev.write('\t')
    fdev.write(listrecTotFileName[i][1])
    fdev.write('\n')
for i in range(1000,len(listrecTotFileName)):
    ftrain.write(listrecTotFileName[i][0])
    ftrain.write('\t')
    ftrain.write(listrecTotFileName[i][1])
    ftrain.write('\n')
ftrain.close()
fdev.close()
ftest.close()