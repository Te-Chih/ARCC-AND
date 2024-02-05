import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
from src.utils import parseJson, saveJson, parse_configion
from src.util_training import setup_seed
import tqdm

# step1: parse config and add parse
config = parse_configion()
setup_seed(config['seed'])
# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]




semantic_emb_bert_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['semantic_emb_bert'])
semantic_emb_w2v_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['semantic_emb_w2v'])
train_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['train_rel_emb_rule'])
valid_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['valid_rel_emb_rule'])
test_rel_emb_rule_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['test_rel_emb_rule'])




all_pid_list_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['all_pid'])
all_pid_to_idx_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['all_pid_to_idx'])
sem_emb_vector_bert_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['sem_emb_vector_bert'])
sem_emb_vector_w2v_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['sem_emb_vector_w2v'])
rel_emb_vector_rule_path =  "{}/{}/{}".format(BasePath,config['processed_path'],config['rel_emb_vector_rule'])



def produce_EmbeddingLayer():
    bert_sem_emb_list = parseJson(semantic_emb_bert_path)
    w2v_sem_emb_list = parseJson(semantic_emb_w2v_path)
    train_rel_emb_list = parseJson(train_rel_emb_rule_path)
    valid_rel_emb_list = parseJson(valid_rel_emb_rule_path)
    test_rel_emb_list = parseJson(test_rel_emb_rule_path)


    all_rel_emb_list_t = dict(train_rel_emb_list, **valid_rel_emb_list)
    all_rel_emb_list = dict(all_rel_emb_list_t, **test_rel_emb_list)

    rel_pid_list = [pid for pid in all_rel_emb_list.keys()]
    print(len(rel_pid_list),len(bert_sem_emb_list))
    pid_to_idx={}
    rel_emb_vector = [0]*len(rel_pid_list)
    sem_emb_vector_bert = [0]*len(rel_pid_list)
    sem_emb_vector_w2v = [0]*len(rel_pid_list)
    pid_list = [0] * len(rel_pid_list)
    for index, r_pid in enumerate(rel_pid_list):
        pid_list[index] = r_pid
        pid_to_idx[r_pid] = index
        rel_emb_vector[index] = all_rel_emb_list[r_pid]
        if config['dataset'] == 'Aminer-18':
            sem_emb_vector_bert[index]= bert_sem_emb_list[r_pid.split("-")[0]]
            sem_emb_vector_w2v[index] = w2v_sem_emb_list[r_pid.split("-")[0]]
        else:
            sem_emb_vector_bert[index] = bert_sem_emb_list[r_pid]
            sem_emb_vector_w2v[index] = w2v_sem_emb_list[r_pid]



    saveJson(all_pid_list_path,pid_list)
    saveJson(all_pid_to_idx_path,pid_to_idx)
    saveJson(sem_emb_vector_bert_path,sem_emb_vector_bert)
    saveJson(sem_emb_vector_w2v_path,sem_emb_vector_w2v)
    saveJson(rel_emb_vector_rule_path,rel_emb_vector)

    print("finish...")




if __name__ == '__main__':
    produce_EmbeddingLayer()
