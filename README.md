# Author Name Disambiguation via Paper Association Refinement and Compositional Contrastive Embedding

This is implementation of our WWWW'24 paper:

Dezhi Liu, Richong Zhang, Junfan Chen, and Xinyue Chen. Author Name Disambiguation via Paper Association Refinement and Compositional Contrastive Embedding.  In Proceedings of the ACM Web Conference 2024 (WWW '24).


## Requirements
- python==3.8
- torch==2.0.1
- install requirements via  pip install -r requirements.txt

## Data

Please download data:

- Aminer-18 https://www.aminer.cn/na-data (na-v2)

- WhoisWho-SND https://www.aminer.cn/billboard/whoiswho   (na-v1: From-Scratch Name Disambiguation )

Unzip the file and Put the different datasets into the corresponding **raw** files.

The data directory is as follows:

    |- ARCC
       |- data
           |- Aminer-18
              |- raw
              |- processed
              |- pretrain_model
           |- WhoisWho-SND
              |- raw
              |- processed
              |- pretrain_model
          
## How to run

####  Data Preprocessing
```
cd $project_path
./process.bash
```

> Provides download of pre-processed data, Unzip the file and Put the different datasets into the corresponding **processed** files.
> pan.baidu.com

#### train and test model
```./run.bash```
> The result is in the output directory