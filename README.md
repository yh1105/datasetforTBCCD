# datasetforTBCCD

There currently include the datasets and the souce code of variants of TBCCD we used in our paper. 
Note that, for bigclonebench, you must need to run python3 ...


The data.zip contains the code fragments of the first 15 questions of the oj dataset, and the bigclonebenchdata.zip is 9134 java code fragments.

The six files of sentenceBCBnoast.zip, sentenceBCBwithid.zip, sentencePOJnoast.zip, sentencePOJwithid.zip, word2vecBCB100noast.zip, word2vecPOJ100noast.zip are prepared for different variants.

(bcb\poj)withidfinetune:tbccd+token,token embeddings are random initialize and tune with training.

(bcb\poj)noastfinetune:tbccd+token-type,token embeddings are random initialize and tune with training.

(bcb\poj)noastnofinetune: tbccd+token-type, token embedding are learned by word2vec and not tune with training. (This variant is not mentioned in the paper, because CDLH is not using astnode information, and uses word2vec to initialize the code, not tune with Training, so we also designed such a variant)

(bcb\poj)noidfinetune: tbccd,token embedding are random initialize and tune with training.

(bcb\poj)newEm: tbccd+token+pace, token embedding are embedded by outr new approach PACE and not tune with training.

(bcb\poj)compareWithCDLH: tbccd+token,token embeddings are random initialize and tune with training. And use 500 code fragment for test set.


You can directly "python3 bcbnewEm.py" or "python pojnewEm.py" to run TBCCD+token+PACE, due to after apply our PACE approach, didn't use other prepare ways.


About how to get the preaper files(such as the six zip files methoded in head), I will put in later.
How to get the train、dev、test dataset by yourself.
1,"python getTrainDevTestDataFileFor(BCB\POJ).py" to get the file.
2,"python getTrainDevTestDataPairFor(BCB\POJ).py" to construct pairs.
3, since the training dataset is very large, you can use "selectPartC.py" or "selectPartJava.py" to random select part training dataset, you can change the parameters in "selectPartC.py" or "selectPartJava.py"
to decide how much training dataset you want tot select.

Note that, for BigCloneBench, there has two .txt file, function.txt and similarity.txt, function.txt contains 9134 code fragment as the same as CDLH, similarity.txt is, 9134*9134, it labels each two code fragment is clone or not clone.


About how to get sentenceBCBnoast.zip, sentenceBCBwithid.zip, sentencePOJnoast.zip, sentencePOJwithid.zip, word2vecBCB100noast.zip, word2vecPOJ100noast.zip, you can see 
getSentenceNoAstnodeBCB.py、getSentenceNoAstnodePOJ.py、getAstSentenceWithIdPOJ.py、getAstSentenceWithIdBCB.py、getWord2V.py

get(*).py are all the prepear work.


If you have any questions, please contact yh0315@pku.edu.cn