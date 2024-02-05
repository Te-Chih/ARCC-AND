[![DOI](https://zenodo.org/badge/751475367.svg)](https://zenodo.org/doi/10.5281/zenodo.10618825)

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

Unzip the file and put the different datasets into the corresponding **data/{dataset}/raw/** directory.

The data directory is as follows:

    |- ARCC-AND
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

####  Step1: Data Preprocessing
```
cd $project_path
./01_process.bash
```

For fast reproduction, you can skip the '01_process.bash' phase and use the data we processed.

Download: https://pan.baidu.com/s/1u3tucycIKdmoA6DMgcEfyA?pwd=edb7 Password: edb7

Unzip the files and place the different files into the **data/{dataset}/processed/** directory  under the corresponding dataset




#### Step2: Training and testing models

```
cd $project_path
./02_model.bash
```

> The result is in the **output** directory

<!--
## Citation
If you find our work useful, please consider citing the following paper.
```

```
-->
