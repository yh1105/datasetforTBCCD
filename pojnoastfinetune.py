import os
import tensorflow as tf
import numpy as np
import network
import pycparser
from sampleC import _traverse_tree_noast
from sampleC import getData_finetune
from sklearn.metrics import precision_score, recall_score, f1_score
from parameters import LEARN_RATE, EPOCHS

def train_model(infile, embeddings, epochs=EPOCHS):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_feats = 100
    nodes_node1, children_node1, nodes_node2, children_node2, res = network.init_net_finetune(
        num_feats,
        embeddingg
    )
    labels_node, loss_node = network.loss_layer(res)

    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)#config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())
    dictt = {}
    listrec = []
    f = open("flistPOJ.txt", 'r')
    line = f.readline().rstrip('\t')
    l = line.split('\t')
    print(len(l))

    for ll in l:
        if not os.path.exists(ll):
            listrec.append(ll)
            continue
        tree = pycparser.parse_file(ll)
        sample, size = _traverse_tree_noast(tree)
        dictt[ll] = sample
    f.close()
    for epoch in range(1, epochs+1):
        f = open(infile, 'r')
        line = "123"
        k = 0
        aaa=0
        while line:
            line = f.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes11,children1,nodes22,children2,batch_labels=getData_finetune(l,dictt,embeddings)
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes11,
                    children_node1: children1,
                    nodes_node2: nodes22,
                    children_node2: children2,
                    labels_node: batch_labels
                }
            )
            if aaa % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', aaa,
                      'Loss:', err,
                      'R:', r,
                      )
            aaa += 1
        f.close()
        correct_labels_dev = []
        predictions_dev = []
        for reci in range(0, 15):
            predictions_dev.append([])
        ff = open("./datasetForVariantsTBCCD/POJ/devdata.txt", 'r')
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            label = l[2]
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes11,children1,nodes22,children2,_=getData_finetune(l,dictt,embeddings)
            k += 1
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes11,
                                  children_node1: children1,
                                  nodes_node2: nodes22,
                                  children_node2: children2,
                              }
                              )
            correct_labels_dev.append(int(label))
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
        ff = open("./datasetForVariantsTBCCD/POJ/testdata.txt", 'r')
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            label = l[2]

            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes11,children1,nodes22,children2,_=getData_finetune(l,dictt,embeddings)
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes11,
                                  children_node1: children1,
                                  nodes_node2: nodes22,
                                  children_node2: children2,
                              }
                              )
            correct_labels_test.append(int(label))
            threaholderr = -0.7+maxstep*0.1
            if output[0] >= threaholderr:
                predictions_test.append(1)
            else:
                predictions_test.append(-1)
        ff.close()
        print("starttest:")
        print("threaholderr:")
        print(threaholderr)
        p = precision_score(correct_labels_test, predictions_test, average='binary')
        r = recall_score(correct_labels_test, predictions_test, average='binary')
        f1score = f1_score(correct_labels_test, predictions_test, average='binary')
        print("recall_test:" + str(r))
        print("precision_test:" + str(p))
        print("f1score_test:" + str(f1score))
if __name__ == '__main__':
    dictt = {}
    dictta = {}
    listta = list()

    feature_size = 100
    fz = open("sentencePOJnoast.txt", 'r')
    line = "123"
    listchar = []
    while line:
        line = fz.readline().rstrip("\n")
        l = line.split(" ")
        listchar.extend(list(set(l)))
        listchar = list(set(listchar))
    fz.close()
    print(listchar)

    for l in listchar:
        listta.append(np.random.normal(0, 0.1, 100).astype(np.float32))
    embeddingg = np.asarray(listta)
    embeddingg = tf.Variable(embeddingg)
    for i in range(len(listchar)):
        dictta[listchar[i]] = i

    train_model('./datasetForVariantsTBCCD/POJ/traindata.txt', dictta)
