import random
f=open("recordtraindataPOJ.txt",'r')
ff=open("precordtraindataPOJ.txt",'w')
while True:
    line=f.readline().rstrip('\n')
    l=line.split('\t')
    if len(l)!=3:
        break
    randum = random.randint(0, 100)
    if randum < 5:
        ff.write(line)
        ff.write('\n')

f.close()
ff.close()
