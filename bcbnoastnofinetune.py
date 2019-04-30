import os
import tensorflow as tf
import network
import javalang
from gensim.models.word2vec import Word2Vec
from sampleJava import _traverse_tree_noast,getData_nofinetune
from parameters import EPOCHS, LEARN_RATE
from sklearn.metrics import precision_score, recall_score, f1_score


def train_model(infile, emmbedding):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_feats = len(emmbedding.wv['AstNode'])
    nodes_node1, children_node1, nodes_node2, children_node2, res = network.init_net_nofinetune(num_feats)
    aaa = 1
    labels_node, loss_node = network.loss_layer(res)
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # ### init the graph
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
        sample, size = _traverse_tree_noast(tree)
        if size > 3000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = sample
    f.close()
    print("wuxiaogeshu:", z)
    for epoch in range(1, EPOCHS + 1):
        f = open(infile, 'r')
        line = "123"
        while line:
            line = f.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break

            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes1, children1, nodes2, children2, batch_labels = getData_nofinetune(l,dictt,emmbedding)
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes1,
                    children_node1: children1,
                    nodes_node2: nodes2,
                    children_node2: children2,
                    labels_node: [batch_labels],
                }
            )
            maxnodes = max(len(nodes1[0]), len(nodes2[0]))
            if aaa % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', aaa,
                      'Loss:', err,
                      'R:', r,
                      'Max nodes:', maxnodes
                      )
            aaa += 1
        correct_labels_dev = []
        predictions_dev = []
        for i in range(0, 15):
            predictions_dev.append([])
        ff = open("./datasetForVariantsTBCCD/BCB/devdata.txt", 'r')
        maxf1value=-1.0
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes1, children1, nodes2, children2, batch_labels = getData_nofinetune(l,dictt,emmbedding)
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  children_node1: children1,
                                  nodes_node2: nodes2,
                                  children_node2: children2,
                              }
                              )
            correct_labels_dev.append(int(l[2]))
            threaholder = -0.7
            for i in range(0, 15):
                if output[0] >= threaholder:
                    predictions_dev[i].append(1)
                else:
                    predictions_dev[i].append(-1)
                threaholder += 0.1
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
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            label = l[2]
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes1, children1, nodes2, children2, batch_labels = getData_nofinetune(l,dictt,emmbedding)
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  children_node1: children1,
                                  nodes_node2: nodes2,
                                  children_node2: children2,
                              }
                              )
            correct_labels_test.append(int(label))
            threaholder = -0.7+maxstep*0.1
            if output[0] >= threaholder:
                predictions_test.append(1)
            else:
                predictions_test.append(-1)
        print("starttest:\n")
        print("threaholder:")
        print(threaholder)
        p = precision_score(correct_labels_test, predictions_test, average='binary')
        r = recall_score(correct_labels_test, predictions_test, average='binary')
        f1score = f1_score(correct_labels_test, predictions_test, average='binary')
        print("recall_test:" + str(r))
        print("precision_test:" + str(p))
        print("f1score_test:" + str(f1score))
        ff.close()

model = Word2Vec.load("word2vecBCB100noast.txt")

train_model('./datasetForVariantsTBCCD/BCB/traindata.txt', model)
