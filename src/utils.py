import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

import argparse
import math
import os
import codecs
import json
import re
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
import torch
import string
import unicodedata
import scipy.sparse as sp
import torch.nn.functional as F
import yaml
import pandas as pd
from src.clusters import paperClusterByDis
from torch_geometric.data import Data,DataLoader as pygDataLoader
# from clusters import paperClusterByDis
from src.featureUtils import *
# from featureUtils import *
# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]

all_letters = string.ascii_letters

stopwords = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
             'is', 'are', 'can', "a", "able", "about", "above", "according", "accordingly", "across", "actually",
             "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
             "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
             "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
             "appreciate", "appropriate", "are", "aren't", "around", "as", "a's", "aside", "ask", "asking",
             "associated", "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes",
             "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides",
             "best", "better", "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant",
             "can't", "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com", "come",
             "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
             "corresponding", "could", "couldn't", "course", "c's", "currently", "definitely", "described", "despite",
             "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards",
             "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially",
             "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly",
             "example", "except", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for",
             "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given",
             "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadn't", "happens", "hardly",
             "has", "hasn't", "have", "haven't", "having", "he", "hello", "help", "hence", "her", "here", "hereafter",
             "hereby", "herein", "here's", "hereupon", "hers", "herself", "he's", "hi", "him", "himself", "his",
             "hither", "hopefully", "how", "howbeit", "however", "i'd", "ie", "if", "ignored", "i'll", "i'm",
             "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
             "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself", "i've", "just",
             "keep", "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly",
             "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks",
             "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover",
             "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
             "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none",
             "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often",
             "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others",
             "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "particular",
             "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
             "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless",
             "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second",
             "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible",
             "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six",
             "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
             "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure",
             "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "that's",
             "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
             "therefore", "therein", "theres", "there's", "thereupon", "these", "they", "they'd", "they'll", "they're",
             "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
             "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries",
             "truly", "try", "trying", "t's", "twice", "two", "un", "under", "unfortunately", "unless", "unlikely",
             "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value",
             "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome",
             "well", "we'll", "went", "were", "we're", "weren't", "we've", "what", "whatever", "what's", "when",
             "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "where's", "whereupon",
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "who's", "whose",
             "why", "will", "willing", "wish", "with", "within", "without", "wonder", "won't", "would", "wouldn't",
             "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",
             "zero", "zt", "ZT", "zz", "ZZ"}



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    ).lower()


def parseJson(path):
    try:
        if os.path.exists(path):
            with codecs.open(path, 'r', 'utf-8') as f:
                jsonObj = json.load(f)
            return jsonObj
        else:
            return None
    except Exception as e:
        print(e)
        return None


def saveJson(path, jsonObj):
    try:
        with codecs.open(path, 'w', 'utf-8') as f:
            f.write(json.dumps(jsonObj, ensure_ascii=False, indent=1))

    except Exception as e:
        print(e)


def formatPaperName(originname):
    name = re.sub('\[.+\]', '', originname)
    name = re.sub('\(.+\)', '', name)
    name = re.sub('\{.+\}', '', name)
    name = re.sub('&quot|&gt|&lt|&amp', '', name)
    name = re.sub('[^\u4e00-\u9fa5a-zA-Z,& -]', '', name)

    if len(name) > 0 and (name[-1] == ',' or name[-1] == '&'):
        name = name[:-1]
    return name


def etl(content):
    if content is None:
        return ''
    if isinstance(content, list):
        content = ' '.join(content)

    content = re.sub('&quot|&gt|&lt|&amp', '', content)
    content = re.sub("[\s+\.\!\/,:;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）\|]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content


def formatName(name):
    name = str(name).lower()
    name = re.sub('[ ,-]', '', name)
    name = re.sub('\[.+\]', '', name)
    return name


def is_Chinese(word):
    for ch in word:
        if not ('\u4e00' <= ch <= '\u9fff'):
            return False
    return True



def evaluate(preds, truths):
    """
    定义：[(<p1,p2>,pre),label]
    pre = 1 <=> pred[p1] == pred[p2]：Predict that p1 and p2 are the same author
    pre = 0 <=> pred[p1] != pred[p2]：Predicts that p1 and p2 are not by the same author
    label = 1 <=> truth[p1] == truth[p2]：The true values p1 and p2 are the same author
    label = 0 <=> truth[p1] != truth[p2]：The true values p1 and p2 are not by the same author.
    TP: [(<p1,p2>,1),1], Prediction that p1 and p2 are the same author and correct (true value p1 and p2 are the same author);
    TN: [(<p1,p2>,0),0]，Prediction that p1 is not the same author as p2 and is correct (true value p1 is not the same author as p2);
    FP: [(<p1,p2>,1),0]，Prediction that p1 and p2 are the same author and wrong (true value p1 and p2 are not the same author);
    FN: [(<p1,p2>,0),0]，Prediction that p1 and p2 are not the same author and are wrongly predicted (true value p1 and p2 are the same author);
    :param preds:
    :param truths:
    :return:
    """
    predMap = {}
    predList = []

    for i, cluster in enumerate(preds):
        predList.extend(cluster)
        for paperId in cluster:
            predMap[paperId] = i

    truthMap = {}
    for talentId in truths.keys():
        for paperId in truths[talentId]:
            # paperId = str(paperId).split('-')[0]
            truthMap[paperId] = talentId

    tp = 0
    fp = 0
    fn = 0
    # tn = 0
    n_samples = len(predList)
    for i in range(n_samples - 1):
        pred_i = predMap[predList[i]]
        for j in range(i + 1, n_samples):
            pred_j = predMap[predList[j]]
            if pred_i == pred_j:
                if truthMap[predList[i]] == truthMap[predList[j]]:
                    tp += 1
                else:
                    fp += 1
            elif truthMap[predList[i]] == truthMap[predList[j]]:
                fn += 1
            # elif truthMap[predList[i]] != truthMap[predList[j]]:
                # tn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn

    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / tp_plus_fp
        recall = tp / tp_plus_fn
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1



def getPaperInfo(paper):
    """
        Using the raw data of a thesis, three sets are obtained;
        1. (collection of clean words for title/abs/venue)
        2. (collection of processed orgs)
        3. (collection of names)

    """
    au_set = set()
    org_set = set()
    common_set = set()

    for word in extract_common_features(paper):
        common_set.add(word)

    for author in paper.get('authors', ''):
        name = author.get('name')
        if len(name) > 0:
            au_set.add(name)
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set, common_set





def getPaperFeatures(paperId, cacheMap, paper_features):
    if paperId in cacheMap:
        return cacheMap[paperId]

    au_set = set()
    t_set = set()
    org_set = set()
    venue_set = set()
    for feature in paper_features:
        if '__NAME__' in feature:
            au_set.add(feature)
        elif '__ORG__' in feature:
            org_set.add(feature)
        elif '__VENUE__' in feature:
            venue_set.add(feature)
        else:
            t_set.add(feature)

    cacheMap[paperId] = (au_set, org_set, t_set, venue_set)

    return au_set, org_set, t_set, venue_set




def similaritySet(strSet1, strSet2):
    mergeCount = 0
    for str1 in strSet1:
        for str2 in strSet2:
            if SequenceMatcher(None, str1, str2).ratio() > 0.9:
                mergeCount += 1
                break
    return mergeCount


def get_config(config_path):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def parse_configion(cfg_path="../config/Aminer-18/cfg.yml"):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--cfg_path', type=str, default=cfg_path,
                        help='path to the config file')
    parser.add_argument('--run_model', type=str, default="debug", choices=['run', 'debug'],
                        help='batch_size')
    dymic_args = vars(parser.parse_args())
    static_args = get_config("{}/{}".format(BasePath,dymic_args['cfg_path']))
    args = dict(dymic_args, **static_args)
    return args


def generate_adj_matrix_by_rulesim(paper_ids, paper_infos, idfMap,
                                   threshold=8):
    paper_num = len(paper_ids)
    cacheMap = {}
    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2
            idf_sum = 0
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f, j)
            mergeAu = len(au_set1 & au_set2)
            mergeOrg = len(org_set1 & org_set2)

            if mergeAu >= 3:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    return rule_sim_matrix


if __name__ == '__main__':
    print(np.argmax([0, 1, 2]))
