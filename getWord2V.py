from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
mySentence=LineSentence('sentenceBCBnoast.txt')#senten.txt
model = Word2Vec(mySentence, size=100, window=5, min_count=1, workers=4)
model.save("word2vecBCB100noast.txt")
