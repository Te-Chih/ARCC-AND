import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from src.utils import *
from src.util_training import setup_seed
import os
import torch
import math


# step1: parse config and add parse
config = parse_configion()
setup_seed(config['seed'])


# Solve the problem of absolute and relative paths for different clients
BasePath = os.path.abspath(os.path.dirname(__file__))
# Current file name
curFileName = os.path.basename(__file__).split('.')[0]
pubs_raw_path =  "{}/{}/{}".format(BasePath,config['raw_path'],config['raw_data'])
bert_courpus_path = "{}/{}/bert_courpus.json".format(BasePath,config['processed_path'])
train_file = "{}/{}/bert_train_courpus.txt".format(BasePath,config['processed_path'])
eval_file =  "{}/{}/bert_eval_courpus.txt".format(BasePath,config['processed_path'])
max_seq_length = 512
train_epoches = 50
batch_size = 24
out_model_path = "{}/{}/bert_model_epoch{}".format(BasePath,config['pretrain_model_path'],train_epoches)
cls_semantic_embedding_bert_path = "{}/{}/bert_cls_semantic_embedding_epoch{}.json".format(BasePath,config['processed_path'],train_epoches)
avg_semantic_embedding_bert_path = "{}/{}/bert_hidden_semantic_embedding_epoch{}.json".format(BasePath,config['processed_path'],train_epoches)



def extract_bert_corpus():
    res = {}
    papers = parseJson(pubs_raw_path)
    for pid, paper in papers.items():
        title = "TITLE: " + str(paper.get("title", " ")).strip()
        abstract = "ABSTRACT: " + str(paper.get("abstract", " ")).strip()
        venue = "VENUE: " + str(paper.get("venue", " ")).strip()
        keywords = paper.get("keywords", " ")
        year = "YEAR: "+str(paper.get("year", " "))
        if title[-1] != '.' :
            title = title + '. '
        else:
            title = title + ' '

        if abstract[-1] != '.':
            abstract = abstract + '. '
        else:
            abstract = abstract + ' '

        if venue[-1] != '.':
            venue = venue + '. '
        else:
            venue = venue + ' '

        if year[-1]  != '.':
            year = year + '. '
        else:
            year = year + ' '

        if isinstance(keywords, list):
            kw = ", ".join(keywords)
            kw = kw.strip()
            kw = kw[:-1] + ". "
            kw = "KEYWORDS: " + kw

        else:
            kw = "KEYWORDS: " + keywords + ". "
            # corpus = title + " " + abstract + " " + venue
        corpus = title + abstract +  kw  + venue + year
        corpus = corpus.strip()
        res[pid] = corpus

    saveJson(bert_courpus_path, res)


def split_train_test():

    bert_courpus = parseJson(bert_courpus_path)
    with open(train_file, "w", encoding="utf-8") as train_f:
        for k, v in list(bert_courpus.items())[:int(0.8 * len(bert_courpus))]:
            train_f.write(str(v) + "\n")
    with open(eval_file, "w", encoding="utf-8") as eval_f:
        for k, v in list(bert_courpus.items())[int(0.8 * len(bert_courpus)):]:
            eval_f.write(str(v) + "\n")


def train_bert_byMLM():

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True,max_length=max_seq_length)

    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=max_seq_length,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=eval_file,
        block_size=max_seq_length,
    )

    training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(out_model_path)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

def get_semantic_embedding_by_bert():

    bert1 = BertModel.from_pretrained(out_model_path, local_files_only=True)
    bert1.cuda()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    bert_embeding = {}
    avg_bert_embeding = {}
    res = parseJson(bert_courpus_path)
    for paperId, sent in res.items():
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids = encoded_dict['input_ids'].cuda()
        attention_mask_ids = encoded_dict['attention_mask'].cuda()

        out = bert1(input_ids=input_ids, attention_mask=attention_mask_ids)
        bert_embeding[paperId] = torch.squeeze(out['pooler_output']).to("cpu").tolist()

    saveJson(cls_semantic_embedding_bert_path, bert_embeding)
    # saveJson(avg_semantic_embedding_bert_path, avg_bert_embeding)


def main():
    extract_bert_corpus() # run only once
    split_train_test() # run only once
    train_bert_byMLM()
    get_semantic_embedding_by_bert()

if __name__ == '__main__':
    main()