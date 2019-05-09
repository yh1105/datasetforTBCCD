import random
f=open("flistBCB.txt",'r')
listt=[]
line=f.readline().rstrip('\t')
l=line.split('\t')
random.shuffle(l)
ff=open("getdatatrainBCB.txt",'w')
ft=open("getdatatestBCB.txt",'w')
fdev=open("getdatadevBCB.txt",'w')
for i in range(1000,len(l)):
    ff.write(l[i])
    ff.write('\t')
for j in range(0,500):
    ft.write(l[j])
    ft.write('\t')

for j in range(500,1000):
    fdev.write(l[j])
    fdev.write('\t')
ff.close()
ft.close()
