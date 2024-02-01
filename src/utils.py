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
import os
import random
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
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
    pre = 1 <=> pred[p1] == pred[p2]：预测p1与p2是同一作者
    pre = 0 <=> pred[p1] != pred[p2]：预测p1与p2不是同一作者;
    label = 1 <=> truth[p1] == truth[p2]：真实值p1与p2是同一作者
    label = 0 <=> truth[p1] != truth[p2]：真实值p1与p2不是同一作者
    TP: [(<p1,p2>,1),1], 预测p1与p2是同一作者且预测对了（真实值p1与p2是同一作者）；
    TN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测对了（真实值p1与p2不是同一作者）；
    FP: [(<p1,p2>,1),0]，预测p1与p2是同一作者且预测错了（真实值p1与p2不是同一作者）；
    FN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测错了（真实值p1与p2是同一作者）；
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

    # 评价方法调整，方便估计单个元素的类簇
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
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set, common_set


def getPaperInfo_noidf(paper):
    """
    用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)

    """
    au_set = set()
    org_set = set()
    # common_set = set()
    #
    # for word in extract_common_features(paper):
    #     common_set.add(word)

    for author in paper.get('authors', ''):
        name = author.get('name')
        if len(name) > 0:
            au_set.add(name)
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set




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


def generate_adj_matrix_by_rule(paper_ids, paper_features, threshold=40):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    edges = []
    values = []
    for i in range(paper_num - 1):
        paperId1, index1 = paper_ids[i].split('-')
        paper_features1 = set(paper_features[paper_ids[i]])
        # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paperId2, index2 = paper_ids[j].split('-')
            paper_features2 = set(paper_features[paper_ids[j]])

            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            if len(au_set1 & au_set2) >= 2:
                edges.append((i, j))
                values.append(1)
            elif len(au_set1 & au_set2) >= 1 and len(org_set1 & org_set2) >= 1:
                edges.append((i, j))
                values.append(1)
            elif len(au_set1 & au_set2) >= 1 or len(org_set1 & org_set2) >= 1:
                if idf_sum >= threshold:
                    edges.append((i, j))
                    values.append(1)

                elif idf_sum >= int(threshold / 2):
                    edges.append((i, j))
                    values.append(idf_sum / threshold)

    edges = np.array(edges, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    weight = sp.coo_matrix((values, (edges[:, 0], edges[:, 1])),
                           shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)
    weight = weight + weight.T.multiply(weight.T > weight) - weight.multiply(weight.T > weight)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, weight


def generate_adj_matrix_by_rule2(paper_ids, paper_features):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    findFather = {}
    label_num = 0
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        if paper_id1 not in findFather:
            findFather[paper_id1] = label_num
            label_num += 1

        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)

            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            matched = False
            if len(au_set1 & au_set2) >= 2:
                matched = True
            elif len(au_set1 & au_set2) >= 1 and len(org_set1 & org_set2) >= 1:
                matched = True
            elif len(au_set1 & au_set2) >= 1 or len(org_set1 & org_set2) >= 1:
                if idf_sum >= 8:
                    matched = True

            if matched:
                if paper_id2 in findFather:
                    label = findFather[paper_id1]
                    for id in findFather:
                        if findFather[id] == label:
                            findFather[id] = findFather[paper_id2]
                else:
                    findFather[paper_id2] = findFather[paper_id1]
    print(set(findFather.values()))

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        for j in range(i + 1, paper_num):
            if paper_ids[i] in findFather and paper_ids[j] in findFather and findFather[paper_ids[i]] == findFather[paper_ids[j]]:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
                continue

    return rule_sim_matrix


def generate_adj_matrix_by_rulesim(paper_ids, paper_infos, idfMap, threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)


    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1


    return rule_sim_matrix


def wiw_generate_adj_matrix_by_rulesim(paper_ids, paper_infos, threshold=8,idfMap_path = 'and/data/aminerEmbeding/wordIdf.json'):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(idfMap_path)

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\n".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix



def generate_adj_matrix_by_rulesim_faster(idfMap,paper_ids, paper_infos, MAXGRAPHNUM,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)


    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\n".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix




def generate_adj_matrix_by_rulesim_v2(paper_ids, paper_infos, threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix


def add_edge_by_semantic_similarity():
    pass


def generate_adj_matrix_by_rulesim_and_semantic_deal_gulidian(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper_embeding_list.append(paper_embedings[paper_id1])
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1
    # 加入语义信息,旨在解决孤立点
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    paper_embeding_tensor = torch.tensor(paper_embeding_list)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    _eye = torch.eye(pred_sim.shape[0])
    pred_sim = pred_sim - _eye
    max_index = torch.argmax(pred_sim, dim=1)
    sim_degree = torch.sum(rule_sim_matrix,dim=1)
    # rule4count = 0
    for row in range(0,max_index.shape[0]):
        # cur_paper_id =  paper_ids[row].split('-')[0]
        # 孤立点
        if sim_degree[row].item() <= 0:
            # 找最相似度最高的节点
            col = int(max_index[row].item())
            rule_sim_matrix[row][col] = 1
            rule_sim_matrix[col][row] = 1
                    # add_edge_by_semantic_similarity(paper_embedings,)
                    # cur_paper_emb =paper_embedings[cur_paper_id]



    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix



def SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, paper_infos, paper_embedings):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)

    #list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float64)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num)
    ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float64)

    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    sim_matrix = 10*sim_matrix
    return sim_matrix, paper_embeding_list
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1


def fix_GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    # au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                # rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                # rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= 8:
                    # rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    # pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
    #                                paper_embeding_tensor.unsqueeze(0), dim=2)
    # semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)
    #
    # sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    # sim_matrix = sim_matrix.type(torch.float64)
    # sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # # sim_matrix = 10*sim_matrix
    # sim_matrix = sim_matrix.type(torch.float32)
    paper_embeding_list = sem_X1_final
    zero_emb = [0.0 for _ in range(len(paper_embeding_list[0]))]
    for i in range(paper_num, MAXGRAPHNUM):
        paper_embeding_list.append(zero_emb)
    # list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list, dtype=torch.float32)
    return rule_sim_matrix, paper_embeding_tensor




def _GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    paper_embeding_list = sem_X1_final
    zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    for i in range(paper_num,MAXGRAPHNUM):
        paper_embeding_list.append(zero_emb)
    #list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix, paper_embeding_tensor
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1







def GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)

    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix, paper_embeding_tensor
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1



def RecurrentGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)

    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1


def nomax_RecurrentGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    # paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)
    paper_embeding_tensor = sem_X1_final
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1



def nomax_impGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    adj_matrix = torch.zeros((paper_num, paper_num),dtype=torch.int64).cuda()
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu
                # adj_matrix[i][j] = 1
                # adj_matrix[i][j] = 1
            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg
                # adj_matrix[i][j] = 1
                # adj_matrix[i][j] = 1


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    # paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)
    paper_embeding_tensor = sem_X1_final
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1





def initial_generate_adj_matrix_by_semantic_guide_graph(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0


    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10
            if cosins_sim_score > 0:
                # 规则1：除同名作者外，有两位相同的作者；
                if mergeAu >= 3:
                    rule1count += 1

                    rule_sim_matrix[i][j] =  mergeAu-1 + cosins_sim_score
                    rule_sim_matrix[j][i] =  mergeAu-1 + cosins_sim_score


                # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
                elif mergeAu >= 2 and mergeOrg >= 1:
                    rule2count += 1
                    rule_sim_matrix[i][j] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                    rule_sim_matrix[j][i] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
                elif mergeAu >= 2 or mergeOrg >= 1:
                    if idf_sum >= threshold:
                        rule3count +=1
                        if mergeAu >= 2:
                            rule_sim_matrix[i][j] = (mergeAu - 1) + cosins_sim_score
                            rule_sim_matrix[j][i] =  (mergeAu - 1)  +  cosins_sim_score
                        else:
                            rule_sim_matrix[i][j] = mergeOrg  +  cosins_sim_score
                            rule_sim_matrix[j][i] = mergeOrg  +  cosins_sim_score
                # 规则4， 利用语义添加高置信度边
                elif cosins_sim_score > 5: #这个可以调节
                      # 添加高置信度的边
                     rule_sim_matrix[i][j] = cosins_sim_score
                     rule_sim_matrix[j][i] = cosins_sim_score

            else:
                pass
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    return rule_sim_matrix, paper_embeding_list





def initial_generate_adj_matrix(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10

            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1

                rule_sim_matrix[i][j] =  1
                rule_sim_matrix[j][i] =  1


            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] =1
                rule_sim_matrix[j][i] =1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    if mergeAu >= 2:
                        rule_sim_matrix[i][j] = 1
                        rule_sim_matrix[j][i] =  1
                    else:
                        rule_sim_matrix[i][j] = 1
                        rule_sim_matrix[j][i] = 1
                # 规则4， 利用语义添加高置信度边
                # elif cosins_sim_score > 5: #这个可以调节
                #       # 添加高置信度的边
                #      rule_sim_matrix[i][j] = cosins_sim_score
                #      rule_sim_matrix[j][i] = cosins_sim_score
    paper_id1 = paper_ids[paper_num-1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    # 加入语义信息,旨在解决孤立点

    # paper_embeding_tensor = torch.tensor(paper_embeding_list)
    # pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
    #                                paper_embeding_tensor.unsqueeze(0), dim=2)
    # _eye = torch.eye(pred_sim.shape[0])
    # pred_sim = pred_sim - _eye
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix,dim=1)
    # # rule4count = 0
    # for row in range(0,max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #
    #         col = int(max_index[row].item())
    #         sim_score=pred_sim[row][col].item() * 10
    #         if sim_score > 2.5: #为孤立点添加一条边。
    #             rule_sim_matrix[row][col] = sim_score
    #             rule_sim_matrix[col][row] = sim_score




    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix, paper_embeding_list


def generate_adj_matrix_by_semantic_guide_graph(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10
            if cosins_sim_score > 0:
                # 规则1：除同名作者外，有两位相同的作者；
                if mergeAu >= 3:
                    rule1count += 1

                    rule_sim_matrix[i][j] =  mergeAu-1 + cosins_sim_score
                    rule_sim_matrix[j][i] =  mergeAu-1 + cosins_sim_score


                # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
                elif mergeAu >= 2 and mergeOrg >= 1:
                    rule2count += 1
                    rule_sim_matrix[i][j] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                    rule_sim_matrix[j][i] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
                elif mergeAu >= 2 or mergeOrg >= 1:
                    if idf_sum >= threshold:
                        rule3count +=1
                        if mergeAu >= 2:
                            rule_sim_matrix[i][j] = (mergeAu - 1) + cosins_sim_score
                            rule_sim_matrix[j][i] =  (mergeAu - 1)  +  cosins_sim_score
                        else:
                            rule_sim_matrix[i][j] = mergeOrg  +  cosins_sim_score
                            rule_sim_matrix[j][i] = mergeOrg  +  cosins_sim_score
                # 规则4， 利用语义添加高置信度边
                elif cosins_sim_score > 5: #这个可以调节
                      # 添加高置信度的边
                     rule_sim_matrix[i][j] = cosins_sim_score
                     rule_sim_matrix[j][i] = cosins_sim_score

            else:
                pass

    return rule_sim_matrix




def generate_adj_matrix_by_rulesim2(paper_ids, paper_features, threshold=20):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            if mergeAu >= 2:
                rule_sim_matrix[i][j] += 1
                rule_sim_matrix[j][i] += 1

            if mergeAu >= 1 and mergeOrg >= 1:
                rule_sim_matrix[i][j] += 1
                rule_sim_matrix[j][i] += 1

            if mergeAu >= 1 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule_sim_matrix[i][j] += 1
                    rule_sim_matrix[j][i] += 1
                else:
                    rule_sim_matrix[i][j] += idf_sum / threshold
                    rule_sim_matrix[j][i] += idf_sum / threshold

    return rule_sim_matrix


def generate_muti_adj_matrix(paper_ids, paper_features, threshold=40):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    rule_sim_matrix1 = torch.zeros((paper_num, paper_num))
    rule_sim_matrix2 = torch.zeros((paper_num, paper_num))
    rule_sim_matrix3 = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            if mergeAu >= 2:
                rule_sim_matrix1[i][j] = 1
                rule_sim_matrix1[j][i] = 1

            if mergeAu >= 1 and mergeOrg >= 1:
                rule_sim_matrix2[i][j] = 1
                rule_sim_matrix2[j][i] = 1

            if idf_sum >= 20:
                rule_sim_matrix3[i][j] = 1
                rule_sim_matrix3[j][i] = 1

    return rule_sim_matrix1, rule_sim_matrix2, rule_sim_matrix3


def generate_samples(pos_matrix, pred_matrix, cosine_matrix, sample_num=10):
    sample_papers = []
    sample_pairwises = []
    labels = []

    for i in range(pos_matrix.shape[0]):
        try:
            neighboors = []
            noneighboors = []
            for j in range(pos_matrix.shape[1]):
                if i != j:
                    if pos_matrix[i][j] == 1:
                        neighboors.append(j)
                    else:
                        noneighboors.append(j)

            if len(neighboors) == 0:
                continue

            # 正样本的采样方式，依照预测的相似度进行采样，预测的相似度越大越容易被采样
            pos_pro = pred_matrix[i, neighboors]
            pos_pro = F.softmax(pos_pro, dim=0).numpy()

            # 负样本的采样方式，依照样本之间的相似度进行采样，样本之间相似度越大越容易被负采样
            neg_pro = cosine_matrix[i, noneighboors]
            neg_pro = F.softmax(neg_pro, dim=0).numpy()

            poss = np.random.choice(neighboors, size=sample_num, p=pos_pro)
            sample_papers.extend([i] * sample_num)
            sample_pairwises.extend(poss)
            labels.extend([1] * sample_num)

            negs = np.random.choice(noneighboors, size=sample_num, p=neg_pro)
            sample_papers.extend([i] * sample_num)
            sample_pairwises.extend(negs)
            labels.extend([0] * sample_num)
        except Exception as e:
            print(e)

    return sample_papers, sample_pairwises, labels


def generate_adj_matrix_by_threshold(paper_ids, paper_features, threshold=40):
    edges = []
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    for i in range(len(paper_ids)):
        paper_features1 = set(paper_features[paper_ids[i]])

        for j in range(i + 1, len(paper_ids)):
            paper_features2 = set(paper_features[paper_ids[j]])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f] if f in idfMap else 0

            if idf_sum >= threshold:
                edges.append([i, j])

    edges = np.array(edges, dtype=np.float32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, adj


def generate_adj_matrix_by_discriminator(emb, dis_model):
    emb = dis_model(emb)
    emb = torch.tensor(emb.detach().numpy(), dtype=torch.float32)
    dot = torch.mm(emb, emb.permute(1, 0))
    max_value = torch.max(dot)
    sim = torch.sigmoid(5 * dot / max_value)
    adj = torch.round(sim - 0.3)
    adj = sp.coo_matrix(adj.detach().numpy())
    return adj


def generate_syntax_adj_matrix(paper_ids, systax_embedings, paper_length):
    adj = [[0 for _ in range(paper_length)] for _ in range(paper_length)]

    for i in range(len(paper_ids)):
        embeding1 = systax_embedings[paper_ids[i]]

        for j in range(i + 1, len(paper_ids)):
            embeding2 = systax_embedings[paper_ids[j]]
            weight = cosine_similarity(embeding1, embeding2)
            adj[i][j] = adj[j][i] = weight

    adj = np.array(adj, dtype=np.float32).reshape((paper_length, paper_length))  # / maxValues
    adj[adj >= 0.8] = 1
    adj[adj < 0.8] = 0
    for i in range(len(paper_ids)):
        adj[i][i] = 1
    return adj

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
    static_args = get_config(dymic_args['cfg_path'])
    args = dict(dymic_args, **static_args)
    return args


if __name__ == '__main__':
    print(np.argmax([0, 1, 2]))
