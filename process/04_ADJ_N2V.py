import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from src.utils import parseJson, saveJson,parse_configion,generate_adj_matrix_by_rulesim
from src.util_training import setup_seed
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
os.environ['PYTHONHASHSEED'] = '0'

config = parse_configion()
setup_seed(config['seed'])

# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]

# 定义要模型产生的输出的目录
wordIdf_path = "{}/{}/wordIdf.json".format(BasePath,config['processed_path'])
pubs_raw_path =  "{}/{}/{}".format(BasePath,config['raw_path'],config['raw_data'])

train_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['train_df'])
valid_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['valid_df'])
test_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['test_df'])

train_adj_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['train_adj_rule'])
valid_adj_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['valid_adj_rule'])
test_adj_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['test_adj_rule'])

train_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['train_rel_emb_rule'])
valid_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['valid_rel_emb_rule'])
test_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['test_rel_emb_rule'])


idfMap = parseJson(wordIdf_path)
paper_infos = parseJson(pubs_raw_path)






def N2V(name,paper_num,rule_sim):
    gn = nx.Graph(label=name)
    gn.add_nodes_from(list(range(paper_num)))
    for i in range(paper_num - 1):
        for j in range(i + 1, paper_num):
            if rule_sim[i][j] > 0:
                gn.add_edge(i, j, weight=rule_sim[i][j])
                gn.add_edge(j, i, weight=rule_sim[i][j])

    n2v = Node2Vec(gn, dimensions=100, walk_length=20, num_walks=10, workers=1)
    model = n2v.fit(window=10, min_count=1, seed=config['seed'])
    return model



def generate_adj_and_n2v():


    # df
    train_data_df = pd.read_csv(train_df_path)
    valid_data_df = pd.read_csv(valid_df_path)
    test_data_df = pd.read_csv(test_df_path)
    # all_data_df = all_data_df.drop('index', axis=1)
    train_name_list = train_data_df['name'].unique().tolist()
    valid_name_list = valid_data_df['name'].unique().tolist()
    test_name_list = test_data_df['name'].unique().tolist()
    # all_name_list = all_data_df['name'].unique().tolist()
    train_adj_matrix_dict = {}
    valid_adj_matrix_dict = {}
    test_adj_matrix_dict = {}
    train_rel_emb_rule = {}
    valid_rel_emb_rule = {}
    test_rel_emb_rule = {}

    for train_name in train_name_list:
        paper_ids=train_data_df[train_data_df['name']==train_name]['paperid'].values.tolist()
        adj_matrix = generate_adj_matrix_by_rulesim(paper_ids,paper_infos,idfMap,threshold=config['idf_threshold'])
        train_adj_matrix_dict[train_name] = adj_matrix.data.tolist()


        model=N2V(train_name,len(paper_ids),adj_matrix)
        paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
        for idx, id in enumerate(paper_ids):
            train_rel_emb_rule[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()

    saveJson(train_adj_rule_path, train_adj_matrix_dict)
    saveJson(train_rel_emb_rule_path, train_rel_emb_rule)


    for eval_name in valid_name_list:
        paper_ids = valid_data_df[valid_data_df['name'] == eval_name]['paperid'].values.tolist()
        adj_matrix = generate_adj_matrix_by_rulesim(paper_ids,paper_infos,idfMap,threshold=config['idf_threshold'])
        valid_adj_matrix_dict[eval_name] = adj_matrix.data.tolist()


        model=N2V(eval_name,len(paper_ids),adj_matrix)
        paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
        for idx, id in enumerate(paper_ids):
            valid_rel_emb_rule[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()

    saveJson(valid_adj_rule_path, valid_adj_matrix_dict)
    saveJson(valid_rel_emb_rule_path, valid_rel_emb_rule)

    for test_name in test_name_list:
        paper_ids=test_data_df[test_data_df['name']==test_name]['paperid'].values.tolist()
        adj_matrix = generate_adj_matrix_by_rulesim(paper_ids,paper_infos,idfMap,threshold=config['idf_threshold'])
        test_adj_matrix_dict[test_name] = adj_matrix.data.tolist()


        model = N2V(test_name, len(paper_ids), adj_matrix)
        paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
        for idx, id in enumerate(paper_ids):
            test_rel_emb_rule[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()

    saveJson(test_adj_rule_path, test_adj_matrix_dict)
    saveJson(test_rel_emb_rule_path, test_rel_emb_rule)





if __name__ == '__main__':
    generate_adj_and_n2v()