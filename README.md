# datasetforTBCCD

There currently include the datasets and the souce code of variants of TBCCD we used in our paper. 



The data.zip contains the code fragments of the first 15 questions of the oj dataset, and the bigclonebenchdata.zip is 9134 java code fragments.

The six files of sentenceBCBnoast.zip, sentenceBCBwithid.zip, sentencePOJnoast.zip, sentencePOJwithid.zip, word2vecBCB100noast.zip, word2vecPOJ100noast.zip are prepared for different variants.

(bcb\poj)withidfinetune:tbccd+token,token embeddings are random initialize and tune with training.

(bcb\poj)noastfinetune:tbccd+token-type,token embeddings are random initialize and tune with training.

(bcb\poj)noastnofinetune: tbccd+token-type, token embedding are learned by word2vec and not tune with training. (This variant is not mentioned in the paper, because CDLH is not using astnode information, and uses word2vec to initialize the code, not tune with Training, so we also designed such a variant)

(bcb\poj)noidfinetune: tbccd,token embedding are random initialize and tune with training.

(bcb\poj)newEm: tbccd+token+pace, token embedding are embedded by outr new approach PACE and not tune with training.

(bcb\poj)compareWithCDLH: tbccd+token,token embeddings are random initialize and tune with training. And use 500 code fragment for test set.


You can directly "python3 bcbnewEm.py" or "python pojnewEm.py" to run TBCCD+token+PACE, due to after apply our PACE approach, didn't use other prepare ways.

You can also unzip other .zip files to run other variants of TBCCD.

About how to get the preaper files(such as the six zip files methoded in head), I will put in later.


If you have any questions, please contact yh0315@pku.edu.cn