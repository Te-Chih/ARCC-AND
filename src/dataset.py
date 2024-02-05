import datetime
import random
from torch.utils.data.dataset import Dataset
from src.util_training import setup_seed
from src.utils import *
from tqdm import tqdm
setup_seed(2021)

class GCN_ContrastiveDataSetSR(Dataset):

    def __init__(self, name_list, data_df):
        """

        :param data_path: Training data paths;
        :param paper_embedings: the semantic vector of the paper.
        :param relation_embedings: the relation vectors for papers;
        """
        super(GCN_ContrastiveDataSetSR, self).__init__()
        self.names = name_list
        self.data_df = data_df
        self.name_paperid , self.name_label = self.__pre_process_df__()

    def __pre_process_df__(self):
        starttime = datetime.datetime.now()

        name_paperid = {}
        name_label = {}
        for ix, row in self.data_df.iterrows():
            name = row['name']
            paperid = row['paperid']
            label = row['label']
            if name not in name_paperid:
                name_paperid[name] = []
                name_paperid[name].append(paperid)
            else:
                name_paperid[name].append(paperid)
            if name not in name_label:
                name_label[name] = []
                name_label[name].append(label)
            else:
                name_label[name].append(label)

        endtime = datetime.datetime.now()
        print("DataSet preprocess df use time:",endtime - starttime)
        return name_paperid, name_label

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        return self.name_paperid[name], self.name_label[name], name
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if len(x) > 0:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    else:
        return 0.5
def transform_textcode(documents, pad_size=512):
    contents = []
    mask_ids = []
    for content in tqdm(documents):
        token = tokenizer.tokenize(content)
        token = [CLS] + token + [SEP]
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                # todo 需要更换截断策略,现在使用的是头部策略，也可以使用尾部策略，目前推荐的方式是头部128，尾部384
                token_ids = token_ids[:pad_size]
                # token_ids = token_ids[:129] + token_ids[-384:-1]
        contents.append(token_ids)
        mask_ids.append(mask)

    return contents, mask_ids

# class TripleDataSet(Dataset):
#
#     def __init__(self, data_path, embeding_path):
#         super(TripleDataSet, self).__init__()
#         self.paperEmbedings = []
#         self.posEmbedings = []
#         self.negEmbedings = []
#         res_train = pd.read_excel(data_path, index=None)
#         papersEmbeding = parseJson(embeding_path)
#         self.len = 0
#         for paperId, posId, negId in res_train.values:
#             self.len += 1
#             self.paperEmbedings.append(np.array(papersEmbeding[paperId], dtype=np.float32))
#             self.posEmbedings.append(np.array(papersEmbeding[posId], dtype=np.float32))
#             self.negEmbedings.append(np.array(papersEmbeding[negId], dtype=np.float32))
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         return self.paperEmbedings[index], self.posEmbedings[index], self.negEmbedings[index]
#
#
# class TupleDataSet(Dataset):
#
#     def __init__(self, data_path, paper_embedings):
#         """
#
#         :param data_path: 训练数据路径；论文对<pid1,pid1,1 | 0> , (paperId, parwiseId, label)
#         :param paper_embedings: 论文的语义embeding
#         """
#         super(TupleDataSet, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         parwiseEmbeding = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         return paperEmbeding, parwiseEmbeding, self.label[index]
#
#
# class GCNTupleData(Dataset):
#
#     def __init__(self, data_path):
#         super(GCNTupleData, self).__init__()
#         res_train = parseJson(data_path)
#         self.samples = None
#         self.nameMap = {}
#         for index, name in enumerate(res_train):
#             if name not in self.nameMap:
#                 self.nameMap[name] = {
#                     'x1': [],
#                     'x2': [],
#                     'y': []
#                 }
#             for item in res_train[name]:
#                 self.nameMap[name]['x1'].append(item[0])
#                 self.nameMap[name]['x2'].append(item[1])
#                 self.nameMap[name]['y'].append(item[2])
#
#     def __len__(self):
#         return len(self.samples['x1'])
#
#     def __getitem__(self, index):
#         return self.samples['x1'][index], self.samples['x2'][index], self.samples['y'][index]
#
#     def change(self, name):
#         self.samples = self.nameMap[name]
#
#
#
#
# class ContrastiveDataSetSR(Dataset):
#
#     def __init__(self, data_path, paper_embedings, relation_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(ContrastiveDataSetSR, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             # try:
#
#             paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[_index].split('-')[0]], dtype=np.float32)
#             relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[_index]], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#             # except Exception as e:
#             #     print(self.paperIds[index],e)
#             X.append([paperEmbeding1, relationEmbeding1])
#             Y.append(self.label_num[_index])
#         # print(len(X),len(Y), index)
#
#         return X, Y
#
#
#
# class WIW_GCN_ContrastiveDataSetSR(Dataset):
#
#     def __init__(self, name_list, data_df):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(WIW_GCN_ContrastiveDataSetSR, self).__init__()
#         # self.papersEmbeding = paper_embedings
#         # self.relationEmbeding = relation_embedings
#         self.names = name_list
#         self.data_df = data_df
#         self.name_paperid , self.name_label = self.__pre_process_df__()
#
#     def __pre_process_df__(self):
#         starttime = datetime.datetime.now()
#
#         name_paperid = {}
#         name_label = {}
#         for ix, row in self.data_df.iterrows():
#             name = row['name']
#             paperid = row['paperid']
#             label = row['label']
#             if name not in name_paperid:
#                 name_paperid[name] = []
#                 name_paperid[name].append(paperid)
#             else:
#                 name_paperid[name].append(paperid)
#             if name not in name_label:
#                 name_label[name] = []
#                 name_label[name].append(label)
#             else:
#                 name_label[name].append(label)
#
#         endtime = datetime.datetime.now()
#         print("DataSet preprocess df use time:",endtime - starttime)
#         return name_paperid,name_label
#
#     def __len__(self):
#         return len(self.names)
#
#     def __getitem__(self, index):
#         name = self.names[index]
#         # sub_df =  self.data_df[self.data_df['name'] == name]
#         # self.index = sub_df['index']
#         # self.paperIds = self.name_paperid[name]
#         # self.tid =  sub_df['tid'].values.tolist()
#         # self.label =  self.name_label[name]
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         return self.name_paperid[name], self.name_label[name],name
#
#
#
#
#
# class NoTrain_ContrastiveDataSetSR(Dataset):
#
#     def __init__(self, data_df, paper_embedings, relation_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(NoTrain_ContrastiveDataSetSR, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         self.paperIds = data_df['paperid'].values.tolist()
#         self.label = data_df['tid'].values.tolist()
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#         relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         return self.paperIds[index], paperEmbeding1, relationEmbeding1
#
#
#
# class ContrastiveDataSetS(Dataset):
#
#     def __init__(self, data_path, paper_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(ContrastiveDataSetS, self).__init__()
#         self.papersEmbeding = paper_embedings
#         # self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             # try:
#
#             paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[_index].split('-')[0]], dtype=np.float32)
#             # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[_index]], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#             # except Exception as e:
#             #     print(self.paperIds[index],e)
#             X.append([paperEmbeding1])
#             Y.append(self.label_num[_index])
#         # print(len(X),len(Y), index)
#
#         return X, Y
#
#
# class SupConLossDataSetR(Dataset):
#
#     def __init__(self, data_path, paper_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(SupConLossDataSetR, self).__init__()
#         self.papersEmbeding = paper_embedings
#         # self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             # try:
#
#             # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[_index].split('-')[0]], dtype=np.float32)
#             paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[_index]], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#             # except Exception as e:
#             #     print(self.paperIds[index],e)
#             X.append([paperEmbeding1])
#             Y.append(self.label_num[_index])
#         # print(len(X),len(Y), index)
#
#         return X, Y
#
#
#
# class aminer12_ContrastiveDataSetSR(Dataset):
#
#     def __init__(self, data_path, paper_embedings, relation_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(aminer12_ContrastiveDataSetSR, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             try:
#
#                 paperEmbeding1 = np.array(self.papersEmbeding[str(self.paperIds[_index])], dtype=np.float32)
#                 relationEmbeding1 = np.array(self.relationEmbeding[str(self.paperIds[_index])], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#                 X.append([paperEmbeding1, relationEmbeding1])
#                 Y.append(self.label_num[_index])
#                 # print(len(X),X[0][0].shape,X[0][1].shape,len(Y), index)
#                 if paperEmbeding1.shape[0] < 1:
#
#                     print(paperEmbeding1.shape, relationEmbeding1.shape, str(self.paperIds[_index]))
#
#             except Exception as e:
#                 print(str(self.paperIds[_index]),e)
#
#
#
#
#         return X, Y
#
#
# class bert_ContrastiveDataSetSR(Dataset):
#
#     def __init__(self, data_path, bert_tokenizer, relation_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(bert_ContrastiveDataSetSR, self).__init__()
#         self.bert_tokenizer = bert_tokenizer
#         self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             # try:
#             # print(self.paperIds[_index].split('-')[0])
#             # print(self.bert_tokenizer[str(self.paperIds[_index].split('-')[0])])
#             # print(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("input_ids"))
#             bert_input_ids = np.array(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("input_ids"), dtype=np.int64)
#             bert_attention_masks = np.array(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("attention_masks"), dtype=np.int64)
#             relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[_index]], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#             # except Exception as e:
#             #     print(self.paperIds[index],e)
#             X.append([bert_input_ids,bert_attention_masks, relationEmbeding1])
#             Y.append(self.label_num[_index])
#         # print(len(X),len(Y), index)
#
#         return X, Y
#
# class bert_ContrastiveDataSet_S(Dataset):
#
#     def __init__(self, data_path, bert_tokenizer):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(bert_ContrastiveDataSet_S, self).__init__()
#         self.bert_tokenizer = bert_tokenizer
#         # self.relationEmbeding = relation_embedings
#         # self.paperIds = []
#         # # self.parwiseIds = []
#         # self.label = []
#         self.res_train = pd.read_csv(data_path)
#         # self.bercorpus  = parseJson("")
#         self.paperIds = self.res_train['paperid'].values.tolist()
#         # self.parwiseIds.append(parwiseId)
#         self.label = self.res_train['tid'].values.tolist()
#         self.label_num = self.res_train['label'].values.tolist()
#         self.name = self.res_train['name'].values.tolist()
#         self.index = self.res_train['index'].values.tolist()
#         print("-----------dataset load---------")
#         self.tid2index,self.paperid2index,self.name2index = self.produce_x2index()
#          # = self.produce_pid2index()
#         print("------dataset init--------")
#
#     def produce_x2index(self):
#         tid2index = {}
#         paperid2index = {}
#         name2index = {}
#
#         for _, row in self.res_train.iterrows():
#             tid = row['tid']
#             paperid = row['paperid']
#             index = row['index']
#             name = row['name']
#             if tid in tid2index.keys():
#                 tid2index[tid].append(index)
#             else:
#                 tid2index[tid] = []
#                 tid2index[tid].append(index)
#             paperid2index[paperid] = [index]
#
#             if name in name2index.keys():
#                 name2index[name].append(index)
#             else:
#                 name2index[name] = []
#                 name2index[name].append(index)
#
#         return tid2index,paperid2index,name2index
#
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # 10对采用
#         # 采样 [zs1,zs1',zs2,zs2',....zs10,zs10']
#         # 获取za1
#
#         sample_num = 8
#         index_list = []
#         index_list.append(index)
#         paperid = self.paperIds[index]
#         name = self.name[index]
#         tid = self.label[index]
#         self_index = self.index[index]
#         assert self_index == index, "index error"
#         # 获取za1'的index
#
#         # 慢
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         #  list(set(a)-set(b))
#         # postive_df = self.res_train[(self.res_train['tid'] == tid) & (self.res_train['paperid'] != paperid)]
#         postive_df = self.res_train.iloc[ list(set(self.tid2index[tid]) - set(self.paperid2index[paperid])) ]
#
#         # hard_negative_df = self.res_train[(self.res_train['name'] == name) & (self.res_train['tid'] != tid)]
#         hard_negative_df = self.res_train.iloc[list(set(self.name2index[name])  - set(self.tid2index[tid])) ]
#
#
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) - set(self.tid2index[tid]) )]
#
#         # easy_negative_df = self.res_train.iloc[list( set(self.index) - set(self.name2index[name]) )]
#
#         # 慢
#         # profiler.stop()
#         # profiler.print()
#
#
#         if postive_df.shape[0] == 0: # 没有正样本对
#             postive_index = [index] # 采用两次本身
#             index_list.extend(postive_index)
#         else:
#             # 从正样本集合中采样1个
#             postive_index_list = postive_df["index"].values.tolist()
#             postive_index = random.sample(postive_index_list, 1)
#             index_list.extend(postive_index)
#
#         # 从hard negative中采样
#
#         # 屏蔽少于2个样本的hard negative sample tid
#
#         # hard_negative_df['tid_num'] = hard_negative_df.groupby('tid')['tid'].transform('count')
#         # hard_negative_df = hard_negative_df[hard_negative_df['tid_num'] >= 2]
#
#         # 得到hard negative id
#         hard_negative_tid_list = list(set(hard_negative_df['tid'].values.tolist()))
#         hard_negative_counter = 0
#         for hard_negative_tid in hard_negative_tid_list:
#             if hard_negative_counter >= sample_num - 1:
#                 break
#             # hard_negative_tid_index_list = hard_negative_df[hard_negative_df["tid"] == hard_negative_tid]['index'].values.tolist()
#             hard_negative_tid_index_list = self.tid2index[hard_negative_tid]
#
#             # print(_tid,_tid_index_list)
#             if len(hard_negative_tid_index_list) >= 2:
#                 hard_negative_tid_index = random.sample(hard_negative_tid_index_list, 2)
#                 index_list.extend(hard_negative_tid_index)
#                 hard_negative_counter += 1
#
#         sample_eazy_num = sample_num - hard_negative_counter - 1
#
#
#         if sample_eazy_num >= 1:
#             # 采用 eazy sample
#             # sample_eazy_num = sample_num - 1 - len(hard_negative_tid_list) #采用数量
#
#             # easy_negative_df['tid_num'] = easy_negative_df.groupby('tid')['tid'].transform('count')
#             # easy_negative_df = easy_negative_df[easy_negative_df['tid_num'] >= 2]
#             # 得到hard negative id
#             easy_negative_df = self.res_train[(self.res_train['name'] != name) & (self.res_train['tid'] != tid)]
#             easy_negative_tid_list = list(set(easy_negative_df['tid'].values.tolist()))
#
#
#             # sample_easy_negative_tid_list = random.sample(easy_negative_tid_list, sample_eazy_num)
#             for easy_tid in easy_negative_tid_list:
#                 # 对每个tid 均采用一对
#                 if sample_eazy_num == 0:
#                     break
#                 # easy_tid_index_list = easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 easy_tid_index_list = self.tid2index[easy_tid]
#                     # easy_negative_df[easy_negative_df["tid"] == easy_tid]['index'].values.tolist()
#                 if len(easy_tid_index_list) >= 2:
#                     easy_tid_index = random.sample(easy_tid_index_list, 2)
#                     index_list.extend(easy_tid_index)
#                     sample_eazy_num -= 1
#
#
#
#         X = []
#         Y = []
#         for  _index in index_list:
#             _index = int(_index)
#
#             # try:
#             # print(self.paperIds[_index].split('-')[0])
#             # print(self.bert_tokenizer[str(self.paperIds[_index].split('-')[0])])
#             # print(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("input_ids"))
#             bert_input_ids = np.array(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("input_ids"), dtype=np.int64)
#             bert_attention_masks = np.array(self.bert_tokenizer[self.paperIds[_index].split('-')[0]].get("attention_masks"), dtype=np.int64)
#             # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[_index]], dtype=np.float32)
#                 # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#                 # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#             # except Exception as e:
#             #     print(self.paperIds[index],e)
#             X.append([bert_input_ids,bert_attention_masks])
#             Y.append(self.label_num[_index])
#         # print(len(X),len(Y), index)
#
#         return X, Y
#
#
#
#
# class TupleDataSetSR(Dataset):
#
#     def __init__(self, data_path, paper_embedings, relation_embedings):
#         """
#
#         :param data_path: 训练数据路径；
#         :param paper_embedings: 论文的语义向量，
#         :param relation_embedings: 论文的关系向量；
#         """
#         super(TupleDataSetSR, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             # if index % 1000 == 0:
#                 # print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         # relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         # paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         # relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         try:
#             paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#             relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#             paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#             relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         except Exception as e:
#             print(self.paperIds[index],e)
#
#         return paperEmbeding1, relationEmbeding1, paperEmbeding2, relationEmbeding2, self.label[index]
#
#
# class TupleDataSetSRT(Dataset):
#
#     def __init__(self, data_path, paper_embedings, relation_embedings, topic_embedings):
#         super(TupleDataSetSRT, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         self.topicEmbeding = topic_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#         relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         topicEmbeding1 = np.array(self.topicEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#         paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#         relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         topicEmbeding2 = np.array(self.topicEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#         return paperEmbeding1, relationEmbeding1, topicEmbeding1, paperEmbeding2, relationEmbeding2, topicEmbeding2, self.label[index]
#
#
# class TupleDataSetSRTBert(Dataset):
#
#     def __init__(self, data_path, paper_codes, paper_masks, relation_embedings, topic_embedings):
#         super(TupleDataSetSRTBert, self).__init__()
#         self.paperCodes = paper_codes
#         self.paperMasks = paper_masks
#         self.relationEmbeding = relation_embedings
#         self.topicEmbeding = topic_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperCode1 = np.array(self.paperCodes[self.paperIds[index].split('-')[0]], dtype=np.int64)
#         paperMask1 = np.array(self.paperMasks[self.paperIds[index].split('-')[0]], dtype=np.int64)
#         relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         topicEmbeding1 = np.array(self.topicEmbeding[self.paperIds[index].split('-')[0]], dtype=np.float32)
#
#         paperCode2 = np.array(self.paperCodes[self.parwiseIds[index].split('-')[0]], dtype=np.int64)
#         paperMask2 = np.array(self.paperMasks[self.parwiseIds[index].split('-')[0]], dtype=np.int64)
#         relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         topicEmbeding2 = np.array(self.topicEmbeding[self.parwiseIds[index].split('-')[0]], dtype=np.float32)
#         return paperCode1, paperMask1, relationEmbeding1, topicEmbeding1, paperCode2, paperMask2, relationEmbeding2, topicEmbeding2, self.label[index]
#
#
# class TupleDataSetGroup(Dataset):
#
#     def __init__(self, data_path, train_map_path, paper_embedings, relation_embedings):
#         super(TupleDataSetGroup, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.relationEmbeding = relation_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         self.res_train = parseJson(data_path)
#         self.test_train_map = parseJson(train_map_path)
#
#     def change(self, name):
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         for train_name in self.test_train_map[name]:
#             for index, (paperId, parwiseId, label) in enumerate(self.res_train[train_name]):
#                 self.paperIds.append(paperId)
#                 self.parwiseIds.append(parwiseId)
#                 self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         relationEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         relationEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         return paperEmbeding1, relationEmbeding1, paperEmbeding2, relationEmbeding2, self.label[index]
#
#
# class TupleDataSetGCN(Dataset):
#
#     def __init__(self, data_path, paper_embedings, gcn1_embedings, gcn2_embedings):
#         super(TupleDataSetGCN, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.gcn1_embedings = gcn1_embedings
#         self.gcn2_embedings = gcn2_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         gcn1Embeding1 = np.array(self.gcn1_embedings[self.paperIds[index]], dtype=np.float32)
#         gcn2Embeding1 = np.array(self.gcn2_embedings[self.paperIds[index]], dtype=np.float32)
#         paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         gcn1Embeding2 = np.array(self.gcn1_embedings[self.parwiseIds[index]], dtype=np.float32)
#         gcn2Embeding2 = np.array(self.gcn2_embedings[self.parwiseIds[index]], dtype=np.float32)
#         return paperEmbeding1, gcn1Embeding1, gcn2Embeding1, paperEmbeding2, gcn1Embeding2, gcn2Embeding2, self.label[index]
#
#
# class TupleDataSetGCNGroup(Dataset):
#
#     def __init__(self, data_path, train_map_path, paper_embedings, gcn1_embedings, gcn2_embedings):
#         super(TupleDataSetGCNGroup, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.gcn1_embedings = gcn1_embedings
#         self.gcn2_embedings = gcn2_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         self.res_train = parseJson(data_path)
#         self.test_train_map = parseJson(train_map_path)
#
#     def change(self, name):
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         for train_name in self.test_train_map[name]:
#             for index, (paperId, parwiseId, label) in enumerate(self.res_train[train_name]):
#                 self.paperIds.append(paperId)
#                 self.parwiseIds.append(parwiseId)
#                 self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding1 = np.array(self.papersEmbeding[self.paperIds[index]], dtype=np.float32)
#         gcn1Embeding1 = np.array(self.gcn1_embedings[self.paperIds[index]], dtype=np.float32)
#         gcn2Embeding1 = np.array(self.gcn2_embedings[self.paperIds[index]], dtype=np.float32)
#         paperEmbeding2 = np.array(self.papersEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         gcn1Embeding2 = np.array(self.gcn1_embedings[self.parwiseIds[index]], dtype=np.float32)
#         gcn2Embeding2 = np.array(self.gcn2_embedings[self.parwiseIds[index]], dtype=np.float32)
#         return paperEmbeding1, gcn1Embeding1, gcn2Embeding1, paperEmbeding2, gcn1Embeding2, gcn2Embeding2, self.label[index]
#
#
# class TupleDataSetSplit(Dataset):
#
#     def __init__(self, data_path, author_emb, org_emb, title_emb):
#         super(TupleDataSetSplit, self).__init__()
#         self.paperEmbedings = []
#         self.parwiseEmbedings = []
#         self.label = []
#         res_train = parseJson(data_path)
#         authorEmbeding = parseJson(author_emb)
#         orgEmbedding = parseJson(org_emb)
#         # venueEmbedding = parseJson(venue_emb)
#         titleEmbedding = parseJson(title_emb)
#         for paperId, parwiseId, label in res_train:
#             paperEmbedding = [authorEmbeding[paperId], orgEmbedding[paperId], titleEmbedding[paperId]]
#             pairpaperEmbedding = [authorEmbeding[parwiseId], orgEmbedding[parwiseId], titleEmbedding[parwiseId]]
#             self.paperEmbedings.append(np.array(paperEmbedding, dtype=np.float32))
#             self.parwiseEmbedings.append(np.array(pairpaperEmbedding, dtype=np.float32))
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         return self.paperEmbedings[index], self.parwiseEmbedings[index], self.label[index]
#
#
# class TupleDataSetSplitRnn(Dataset):
#
#     def __init__(self, data_path, allSplitEmbedding):
#         super(TupleDataSetSplitRnn, self).__init__()
#         self.embedding = allSplitEmbedding
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         nameEmbeding, orgEmbeding, titleEmbeding = self.embedding[self.paperIds[index]]
#         nameEmbeding = np.array(nameEmbeding, dtype=np.float32)
#         orgEmbeding = np.array(orgEmbeding, dtype=np.float32)
#         titleEmbeding = np.array(titleEmbeding, dtype=np.float32)
#         pNameEmbeding, pOrgEmbeding, pTitleEmbeding = self.embedding[self.parwiseIds[index]]
#         pNameEmbeding = np.array(pNameEmbeding, dtype=np.float32)
#         pOrgEmbeding = np.array(pOrgEmbeding, dtype=np.float32)
#         pTitleEmbeding = np.array(pTitleEmbeding, dtype=np.float32)
#         return nameEmbeding, orgEmbeding, titleEmbeding, pNameEmbeding, pOrgEmbeding, pTitleEmbeding, self.label[index]
#
#
# class TupleDataSetSplitGRU(Dataset):
#
#     def __init__(self, data_path, allSplitEmbedding):
#         super(TupleDataSetSplitGRU, self).__init__()
#         self.embedding = allSplitEmbedding
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         nameEmbeding, orgEmbeding, titleEmbeding = self.embedding[self.paperIds[index]]
#         nameEmbeding = np.array(nameEmbeding, dtype=np.float32)
#         orgEmbeding = np.array(orgEmbeding, dtype=np.float32)
#         titleEmbeding = np.array(titleEmbeding, dtype=np.float32)
#         pNameEmbeding, pOrgEmbeding, pTitleEmbeding = self.embedding[self.parwiseIds[index]]
#         pNameEmbeding = np.array(pNameEmbeding, dtype=np.float32)
#         pOrgEmbeding = np.array(pOrgEmbeding, dtype=np.float32)
#         pTitleEmbeding = np.array(pTitleEmbeding, dtype=np.float32)
#         return nameEmbeding, orgEmbeding, titleEmbeding, pNameEmbeding, pOrgEmbeding, pTitleEmbeding, self.label[index]
#
#
# class TupleDataSetSplitGRUSR(Dataset):
#
#     def __init__(self, data_path, allSplitEmbedding, relation_embedings):
#         super(TupleDataSetSplitGRUSR, self).__init__()
#         self.embedding = allSplitEmbedding
#         self.relationEmbeding = relation_embedings
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         nameEmbeding, orgEmbeding, titleEmbeding = self.embedding[self.paperIds[index]]
#         nameEmbeding = np.array(nameEmbeding, dtype=np.float32)
#         orgEmbeding = np.array(orgEmbeding, dtype=np.float32)
#         titleEmbeding = np.array(titleEmbeding, dtype=np.float32)
#         reEmbeding1 = np.array(self.relationEmbeding[self.paperIds[index]], dtype=np.float32)
#         pNameEmbeding, pOrgEmbeding, pTitleEmbeding = self.embedding[self.parwiseIds[index]]
#         pNameEmbeding = np.array(pNameEmbeding, dtype=np.float32)
#         pOrgEmbeding = np.array(pOrgEmbeding, dtype=np.float32)
#         pTitleEmbeding = np.array(pTitleEmbeding, dtype=np.float32)
#         reEmbeding2 = np.array(self.relationEmbeding[self.parwiseIds[index]], dtype=np.float32)
#         return nameEmbeding, orgEmbeding, titleEmbeding, reEmbeding1, pNameEmbeding, pOrgEmbeding, pTitleEmbeding, reEmbeding2, self.label[index]
#
#
# class TupleDataSetCnn(Dataset):
#
#     def __init__(self, data_path, allSplitEmbedding):
#         super(TupleDataSetCnn, self).__init__()
#         self.embedding = allSplitEmbedding
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         paperEmbeding = self.embedding[self.paperIds[index]]
#         paperEmbeding = np.array(paperEmbeding, dtype=np.float32)
#         pairwiseEmbeding = self.embedding[self.parwiseIds[index]]
#         pairwiseEmbeding = np.array(pairwiseEmbeding, dtype=np.float32)
#         return paperEmbeding, pairwiseEmbeding, self.label[index]
#
#
# class TupleDataSetSplitRnn2(Dataset):
#
#     def __init__(self, data_path, allSplitEmbedding, allPaperEmbedding):
#         super(TupleDataSetSplitRnn2, self).__init__()
#         self.allSplitEmbedding = allSplitEmbedding
#         self.allPaperEmbedding = allPaperEmbedding
#         self.paperIds = []
#         self.parwiseIds = []
#         self.label = []
#         res_train = parseJson(data_path)
#
#         for index, (paperId, parwiseId, label) in enumerate(res_train):
#             if index % 1000 == 0:
#                 print("loading :", index)
#
#             self.paperIds.append(paperId)
#             self.parwiseIds.append(parwiseId)
#             self.label.append(int(label))
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         nameEmbeding, orgEmbeding, titleEmbeding = self.allSplitEmbedding[self.paperIds[index]]
#         paperEmbedding = self.allPaperEmbedding[self.paperIds[index]]
#         nameEmbeding = np.array(nameEmbeding, dtype=np.float32)
#         orgEmbeding = np.array(orgEmbeding, dtype=np.float32)
#         titleEmbeding = np.array(titleEmbeding, dtype=np.float32)
#         paperEmbedding = np.array(paperEmbedding, dtype=np.float32)
#         pNameEmbeding, pOrgEmbeding, pTitleEmbeding = self.allSplitEmbedding[self.parwiseIds[index]]
#         pairwiseEmbedding = self.allPaperEmbedding[self.parwiseIds[index]]
#         pNameEmbeding = np.array(pNameEmbeding, dtype=np.float32)
#         pOrgEmbeding = np.array(pOrgEmbeding, dtype=np.float32)
#         pTitleEmbeding = np.array(pTitleEmbeding, dtype=np.float32)
#         pairwiseEmbedding = np.array(pairwiseEmbedding, dtype=np.float32)
#         return nameEmbeding, orgEmbeding, titleEmbeding, paperEmbedding, pNameEmbeding, pOrgEmbeding, pTitleEmbeding, pairwiseEmbedding, self.label[index]
#
#
# class LDAVAEDataSet(Dataset):
#
#     def __init__(self, train_datas, paper_embedings, vocab_size=100):
#         super(LDAVAEDataSet, self).__init__()
#         self.papersEmbeding = paper_embedings
#         self.vocab_size = vocab_size
#         self.res_train = train_datas
#
#     def __len__(self):
#         return len(self.res_train)
#
#     def __getitem__(self, index):
#         paperEmbeding = [0] * self.vocab_size
#         for x in self.papersEmbeding[self.res_train[index]]:
#             paperEmbeding[int(x[0])] = x[1]
#
#         paperEmbeding = np.array(paperEmbeding, dtype=np.float32)
#         return paperEmbeding
#
#
# class TupleDataSetBert(Dataset):
#
#     def __init__(self, data_path, file_path):
#         super(TupleDataSetBert, self).__init__()
#         self.paperContents = []
#         self.parwiseContents = []
#         self.label = []
#         res_train = parseJson(data_path)
#         papers = parseJson(file_path)
#         for paperId, parwiseId, label in res_train[:1000]:
#             paperId = paperId.split('-')[0]
#             parwiseId = parwiseId.split('-')[0]
#             self.paperContents.append(papers[paperId]["title"])
#             self.parwiseContents.append(papers[parwiseId]["title"])
#             self.label.append(int(label))
#
#         self.paperCodes, self.paperMask = transform_textcode(self.paperContents, pad_size=64)
#         self.paperCodes = np.array(self.paperCodes, np.int64)
#         self.paperMask = np.array(self.paperMask, np.int64)
#
#         self.parwiseCodes, self.parwiseMask = transform_textcode(self.parwiseContents, pad_size=64)
#         self.parwiseCodes = np.array(self.parwiseCodes, np.int64)
#         self.parwiseMask = np.array(self.parwiseMask, np.int64)
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         return self.paperCodes[index], self.paperMask[index], self.parwiseCodes[index], self.parwiseMask[index], self.label[index]


if __name__ == '__main__':
    pass
