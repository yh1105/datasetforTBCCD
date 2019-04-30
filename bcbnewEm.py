import os
import tensorflow as tf
import numpy as np
import network
from sampleJava import getData_nofinetune,_traverse_treewithid
import javalang
from parameters import EPOCHS, LEARN_RATE
from sklearn.metrics import precision_score, recall_score, f1_score
def getWordEmd(word):
    listrechar = np.array([0.0 for i in range(0, len(listchar))])
    tt = 1
    for lchar in word:
        listrechar += np.array(((len(word) - tt + 1) * 1.0 / len(word)) * np.array(dicttChar[lchar]))
        tt += 1
    return listrechar
def train_model(infile, embeddings):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_feats = len(getWordEmd('ForStatement'))
    nodes_node1, children_node1, nodes_node2, children_node2, res = network.init_net_nofinetune(num_feats)
    labels_node, loss_node = network.loss_layer(res)
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  # config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())
    dictt = {}
    listrec = []
    f = open("flistBCB.txt", 'r')
    line = f.readline().rstrip('\t')
    l = line.split('\t')
    z = 0
    for ll in l:
        if not os.path.exists(ll):
            listrec.append(ll)
            continue
        faa = open(ll, 'r', encoding="utf-8")
        fff = faa.read()
        tree = javalang.parse.parse_member_signature(fff)
        sample, size = _traverse_treewithid(tree)
        if size > 3000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = sample
    f.close()
    for epoch in range(1, EPOCHS + 1):
        f = open(infile, 'r')
        line = "123"
        k = 0
        while line:
            line = f.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, children1, nodes2, children2, la=getData_nofinetune(l,dictt,embeddings)
            batch_labels.append(la)
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes1,
                    children_node1: children1,
                    nodes_node2: nodes2,
                    children_node2: children2,
                    labels_node: batch_labels
                }
            )
            maxnodes = max(len(nodes1[0]), len(nodes2[0]))
            if k % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', k,
                      'Loss:', err,
                      'R:', r,
                      'Max nodes:', maxnodes
                      )
        f.close()
        correct_labels_dev = []
        predictions_dev = []
        for reci in range(0, 15):
            predictions_dev.append([])
        ff = open("./datasetForVariantsTBCCD/BCB/devdata.txt", 'r')
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, children1, nodes2, children2, la = getData_nofinetune(l, dictt, embeddings)
            batch_labels.append(la)
            k += 1
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  children_node1: children1,
                                  nodes_node2: nodes2,
                                  children_node2: children2,
                              }
                              )
            correct_labels_dev.append(int(batch_labels[0]))
            threaholder = -0.7
            for i in range(0, 15):
                if output[0] >= threaholder:
                    predictions_dev[i].append(1)
                else:
                    predictions_dev[i].append(-1)
                threaholder += 0.1
        maxstep = 0
        maxf1value = 0
        for i in range(0, 15):
            f1score = f1_score(correct_labels_dev, predictions_dev[i], average='binary')
            if f1score > maxf1value:
                maxf1value = f1score
                maxstep = i
        ff.close()
        correct_labels_test = []
        predictions_test = []
        ff = open("./datasetForVariantsTBCCD/BCB/testdata.txt", 'r')
        line = "123"
        k = 0
        print("starttest:")
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, children1, nodes2, children2, la = getData_nofinetune(l, dictt, embeddings)
            batch_labels.append(la)
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  children_node1: children1,
                                  nodes_node2: nodes2,
                                  children_node2: children2,
                              }
                              )
            k += 1
            correct_labels_test.append(int(batch_labels[0]))
            threaholderr = -0.7+maxstep*0.1
            if output[0] >= threaholderr:
                predictions_test.append(1)
            else:
                predictions_test.append(-1)
        ff.close()
        print("testdata\n")
        print("threa:")
        print(threaholderr)
        p = precision_score(correct_labels_test, predictions_test, average='binary')
        r = recall_score(correct_labels_test, predictions_test, average='binary')
        f1score = f1_score(correct_labels_test, predictions_test, average='binary')
        print("recall_test:" + str(r))
        print("precision_test:" + str(p))
        print("f1score_test:" + str(f1score))
        ff.close()

def dfsDict(root):
    global listtfinal
    listtfinal.append(str(root['node']))
    global numnodes
    numnodes+=1
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt)
if __name__ == '__main__':
    f = open("sentenceBCBwithid.txt", 'r')
    line = "123"
    listword = []
    while line:
        line = f.readline().rstrip("\n")
        listt = line.split(" ")
        listword.extend(listt)
        listword = list(set(listword))
    f.close()
    #print(len(listword))
    dicttChar = {}
    def _onehot(i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]
    listchar = ['7', 'I', 'E', 'D', 'u', 'C', 'Y', 'W', 'y', '|', '9', '^', 'X', 't', 'a', 'o', 'Z', 'b', 'A', 'J', 'R',
                'w', '?', 'g', '3', '$', 'B', 'l', '5', 'z', 'v', 'T', '2', 'd', '<', 'e', 'M', 'c', 'S', 'm', '4', 'K',
                'O', 'f', 'i', '=', 'Q', '+', 'x', 'N', '1', 'r', 'p', 'G', 'k', '*', 'q', 'L', 'P', '.', 'n', 'j', 'V',
                'U', '6', '/', '%', '8', 'F', 's', '!', '-', '&', '>', 'h', 'H', '0', '_']
    for i in range(0, len(listchar)):
        dicttChar[listchar[i]] = _onehot(i, len(listchar))
    dictfinalem = {}
    t = 0
    for l in listword:
        t += 1
        dictfinalem[l] = getWordEmd(l)
    train_model('./datasetForVariantsTBCCD/BCB/traindata.txt', dictfinalem)