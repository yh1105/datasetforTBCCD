import random
ftr=open("getdatatrainPOJ.txt",'r')
ftd=open("getdatadevPOJ.txt",'r')
fte=open("getdatatestPOJ.txt",'r')
frtr=open("recordtraindataPOJ.txt",'w')
frtd=open("recordtraindevPOJ.txt",'w')
frte=open("recordtraintestPOJ.txt",'w')
line="123"
listrec=[]
listtemp=[]
while line:
    line=ftr.readline().rstrip('\n')
    l=line.split('\t')
    if len(l)!=2:
        break
    listtemp.append(l[0])
    listtemp.append(l[1])
    listrec.append(listtemp)
    listtemp=[]
print(len(listrec))
listreee=[]
for i in range(0,len(listrec)-1):
    for j in range(i+1,len(listrec)):
        listtempp=[]
        if listrec[i][1]==listrec[j][1]:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("1")
        else:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("-1")
        listreee.append(listtempp)
random.shuffle(listreee)
for listt in listreee:
    frtr.write(listt[0])
    frtr.write('\t')
    frtr.write(listt[1])
    frtr.write('\t')
    frtr.write(listt[2])
    frtr.write('\n')

line="123"
listrec=[]
listtemp=[]
while line:
    line=ftd.readline().rstrip('\n')
    l=line.split('\t')
    if len(l)!=2:
        break
    listtemp.append(l[0])
    listtemp.append(l[1])
    listrec.append(listtemp)
    listtemp=[]
print(len(listrec))
listreee=[]
for i in range(0,len(listrec)-1):
    for j in range(i+1,len(listrec)):
        listtempp=[]
        if listrec[i][1]==listrec[j][1]:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("1")
        else:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("-1")
        listreee.append(listtempp)
random.shuffle(listreee)
for listt in listreee:
    frtd.write(listt[0])
    frtd.write('\t')
    frtd.write(listt[1])
    frtd.write('\t')
    frtd.write(listt[2])
    frtd.write('\n')
line="123"
listrec=[]
listtemp=[]
while line:
    line=fte.readline().rstrip('\n')
    l=line.split('\t')
    if len(l)!=2:
        break
    listtemp.append(l[0])
    listtemp.append(l[1])
    listrec.append(listtemp)
    listtemp=[]
print(len(listrec))
listreee=[]
for i in range(0,len(listrec)-1):
    for j in range(i+1,len(listrec)):
        listtempp=[]
        if listrec[i][1]==listrec[j][1]:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("1")
        else:
            listtempp.append(listrec[i][0])
            listtempp.append(listrec[j][0])
            listtempp.append("-1")
        listreee.append(listtempp)
random.shuffle(listreee)
for listt in listreee:
    frte.write(listt[0])
    frte.write('\t')
    frte.write(listt[1])
    frte.write('\t')
    frte.write(listt[2])
    frte.write('\n')
frtr.close()
ftr.close()
ftd.close()
fte.close()
frte.close()
frtd.close()
