import random
import os

ft=open("functions.txt",'r')
os.chdir("./bigclonebenchdata")
line="123"
k=0
t=0
isw=False
filename="123"
dictt={}
while True:
    line=ft.readline()
    l=line.split('\t')
    if l[0]=="FUNCTION_ID:":
        aaa=l[1].rstrip('\n').rstrip('\r')
        #filename=aaa+".txt"
        #ftempwrite=open(filename,"w")
        bbb="./bigclonebenchdata/"+aaa+".txt"
        dictt[bbb]=t
        t+=1
        if t==9134:
            break
    # else:
    #     ftempwrite.write(line)
print(dictt)
print (len(dictt))
ft.close()
os.chdir("/home/yuhao/finalproject")
ftrain=open("recordtraindataBCB.txt",'w')
fdev=open("recorddevdataBCB.txt",'w')
ftest=open("recordtestdataBCB.txt",'w')
fff=open("getdatatrainBCB.txt",'r')
ffd=open("getdatadevBCB.txt",'r')
ftt=open("getdatatestBCB.txt",'r')
f=open("similarity.txt",'r')
line="123"
k=0
rec=[]
bt1=0
bf1=0
z=0
recc=[]
while True:
    line = f.readline()
    #print line
    l = line.split(' ')
    if len(l)==1:
        break
    lis=[]
    for i in range(0,len(l)):
        if l[i].startswith("0") or l[i].startswith("-1"):
            bf1 += 1
            lis.append("-1")
        else:
            bt1 += 1
            lis.append("1")
    recc.append(lis)
    z += 1
print("bt1:",bt1)
print("bf1:",bf1)
line=fff.readline().rstrip('\t')
listrectrain=line.split('\t')
lll=[]
for i in range(0,len(listrectrain)-1):
    for j in range(i+1,len(listrectrain)):
        llll=[]
        llll.append(listrectrain[i])
        llll.append(listrectrain[j])
        llll.append(recc[dictt[listrectrain[i]]][dictt[listrectrain[j]]])
        lll.append(llll)
random.shuffle(lll)
for lllll in lll:
    ftrain.write(lllll[0])
    ftrain.write('\t')
    ftrain.write(lllll[1])
    ftrain.write('\t')
    ftrain.write(lllll[2])
    ftrain.write('\n')
ftrain.close()

line=ffd.readline().rstrip('\t')
listrecdev=line.split('\t')
lll=[]
for i in range(0,len(listrecdev)-1):
    for j in range(i+1,len(listrecdev)):
        llll = []
        llll.append(listrecdev[i])
        llll.append(listrecdev[j])
        llll.append(recc[dictt[listrecdev[i]]][dictt[listrecdev[j]]])
        lll.append(llll)
random.shuffle(lll)
for lllll in lll:
    fdev.write(lllll[0])
    fdev.write('\t')
    fdev.write(lllll[1])
    fdev.write('\t')
    fdev.write(lllll[2])
    fdev.write('\n')
fdev.close()

line=ftt.readline().rstrip('\t')
listrectest=line.split('\t')
lll=[]
for i in range(0,len(listrectest)-1):
    for j in range(i+1,len(listrectest)):
        llll = []
        llll.append(listrectest[i])
        llll.append(listrectest[j])
        llll.append(recc[dictt[listrectest[i]]][dictt[listrectest[j]]])
        lll.append(llll)
random.shuffle(lll)
for lllll in lll:
    ftest.write(lllll[0])
    ftest.write('\t')
    ftest.write(lllll[1])
    ftest.write('\t')
    ftest.write(lllll[2])
    ftest.write('\n')
ftest.close()
