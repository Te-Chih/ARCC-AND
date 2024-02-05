import os
import random
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from tqdm import tqdm
from gensim.models import Word2Vec, Doc2Vec
from collections import defaultdict
from src.util_training import setup_seed
from src.utils import parse_configion,parseJson,saveJson,extract_common_features
import numpy as np




config = parse_configion()
setup_seed(config['seed'])
# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]
idf_courpus_path = "{}/{}/idf_corpus.json".format(BasePath,config['processed_path'])

pubs_raw_path =  "{}/{}/{}".format(BasePath,config['raw_path'],config['raw_data'])
wordIdf_path = "{}/{}/wordIdf.json".format(BasePath,config['processed_path'])

# w2v semantic embedding
w2v_model_path = "{}/{}/w2v_semantic.model".format(BasePath,config['pretrain_model_path'])
semantic_emb_w2v = "{}/{}/{}".format(BasePath,config['processed_path'],config['semantic_emb_w2v'])


paper_infos = parseJson(pubs_raw_path)

def getAllPapersFeatures():
    # paper_feature_list = {}
    semantic_corpus = []

    # Get the title, abstract, and venue of the paper.
    for i, pid in enumerate(paper_infos):
        # if i % 1000:
        #     print(i)
        paper = paper_infos[pid]
        paper_features = extract_common_features(paper)
        semantic_corpus.append(paper_features)
        # paper_feature_list[pid] = paper_features
    saveJson(idf_courpus_path, semantic_corpus)
    # saveJson(paper_feature_path, paper_feature_list)



def calWordIdf():
    cropus = parseJson(idf_courpus_path)
    idfMap = defaultdict(int)
    docNum = len(cropus)

    for doc in tqdm(cropus, desc='cal word idf'):
        wordSet = set(doc)
        for word in wordSet:
            idfMap[word] += 1

    for word in idfMap:
        idfMap[word] = np.log(docNum / idfMap[word])

    saveJson(wordIdf_path, idfMap)




def trainWord2Vec(dim=100):
    corpus = parseJson(idf_courpus_path)
    data = []
    for author_feature in corpus:
        random.shuffle(author_feature)
        data.append(author_feature)

    # model = Word2Vec(data, size=dim, window=5, min_count=5, workers=20)
    model = Word2Vec(data, size=dim, window=5, min_count=5, workers=20)
    model.save(w2v_model_path)

#
def getAllPapersEmbedding():
    corpus = parseJson(idf_courpus_path)
    wordIdf = parseJson(wordIdf_path)
    model = Word2Vec.load(w2v_model_path)  # 加载词向量

    allPapersEmbeding = {}

    for paperId, feature in tqdm(zip(paper_infos.keys(), corpus), desc="cal paper embedding"):
        vectors = []
        sumIdf = 0

        for word in feature:
            try:
                wordEmbeding = model.wv[word]
                idf = wordIdf[word]
                sumIdf += idf
                vectors.append(wordEmbeding * idf)
            except:
                pass

        allPapersEmbeding[paperId] = (np.sum(vectors, axis=0)/sumIdf).tolist()

    saveJson(semantic_emb_w2v, allPapersEmbeding)


if __name__ == '__main__':
    getAllPapersFeatures()  # Extracting paper features to build a corpus
    calWordIdf()  # Calculate the inverse text frequency IDF of the word
    trainWord2Vec()
    getAllPapersEmbedding()




