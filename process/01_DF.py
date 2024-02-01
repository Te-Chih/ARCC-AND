import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
from src.utils import parseJson, saveJson, parse_configion
from src.util_training import setup_seed
import pandas as pd

# step1: parse config and add parse
config = parse_configion()
setup_seed(config['seed'])
# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]

train_raw_data_path = "{}/{}/{}".format(BasePath,config['raw_path'],config['train_raw_data'])
valid_raw_data_path = "{}/{}/{}".format(BasePath,config['raw_path'],config['valid_raw_data'])

test_raw_data_path = "{}/{}/{}".format(BasePath,config['raw_path'],config['test_raw_data'])

all_pid2name_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['all_pid2name'])
train_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['train_df'])
valid_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['valid_df'])
test_df_path = "{}/{}/{}".format(BasePath,config['processed_path'],config['test_df'])


train_dataset = parseJson(train_raw_data_path)
valid_dataset = parseJson(valid_raw_data_path)
test_dataset = parseJson(test_raw_data_path)

def generate_train_validation_dataframe():
    paper_labeled = []
    tid2label = {}
    counter = 0
    for name, data in train_dataset.items():
        for tid in data.keys():
            if tid not in tid2label:
                tid2label[tid] = counter
                counter+=1


    for name, data in train_dataset.items():
        for tid, paper_list in data.items():
            for paper in paper_list:
                dic = {
                    "paperid":paper,
                    "name": name,
                    "tid": tid,
                    "label": tid2label[tid]
                }
                paper_labeled.append(dic)
    df = pd.DataFrame(paper_labeled)
    df1 = df.sample(frac=0.8)  # Random sampling at 0.8
    df2 = df[~df.index.isin(df1.index)]
    df1['index'] =  range(len(df1))
    df2['index'] =  range(len(df2))
    df1.to_csv(train_df_path,index=False)
    df2.to_csv(valid_df_path,index=False)


def generate_train_dataframe():

    paper_labeled = []
    index = 0
    tid2label = {}
    counter = 0
    for name, data in train_dataset.items():
        for tid in data.keys():
            if tid not in tid2label:
                tid2label[tid] = counter
                counter += 1
    for name, data in train_dataset.items():
        for tid, paper_list in data.items():

            for paper in paper_list:
                dic = {
                    "index":index,
                    "paperid":paper,
                    "name": name,
                    "tid": tid,
                    "label": tid2label[tid]
                }
                index+=1
                paper_labeled.append(dic)
    df = pd.DataFrame(paper_labeled)
    df.to_csv(train_df_path,index=False)


def generate_valid_dataframe():

    paper_labeled = []
    index = 0
    tid2label = {}
    counter = 0
    for name, data in valid_dataset.items():
        for tid in data.keys():
            if tid not in tid2label:
                tid2label[tid] = counter
                counter += 1
    for name, data in valid_dataset.items():
        for tid, paper_list in data.items():

            for paper in paper_list:
                dic = {
                    "index":index,
                    "paperid":paper,
                    "name": name,
                    "tid": tid,
                    "label": tid2label[tid]
                }
                index+=1
                paper_labeled.append(dic)
    df = pd.DataFrame(paper_labeled)
    df.to_csv(valid_df_path,index=False)


def generate_test_dataframe():

    paper_labeled = []
    index = 0
    tid2label = {}
    counter = 0
    for name, data in test_dataset.items():
        for tid in data.keys():
            if tid not in tid2label:
                tid2label[tid] = counter
                counter += 1
    for name, data in test_dataset.items():
        for tid, paper_list in data.items():

            for paper in paper_list:
                dic = {
                    "index":index,
                    "paperid":paper,
                    "name": name,
                    "tid": tid,
                    "label": tid2label[tid]
                }
                index+=1
                paper_labeled.append(dic)
    df = pd.DataFrame(paper_labeled)
    df.to_csv(test_df_path,index=False)




if __name__ == '__main__':
    if config['dataset'] == 'Aminer-18':
        generate_train_validation_dataframe()
    else:
        generate_train_dataframe()
        generate_valid_dataframe()

    generate_test_dataframe()