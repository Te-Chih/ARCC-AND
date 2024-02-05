#!/bin/bash
echo Directory is $PWD
echo ${SLURM_JOB_NODELIST}

# Change to your own runtime environment
source /home/LAB/liudezhi/anaconda3/etc/profile.d/conda.sh
conda activate ARCC

cfg_path='../config/Aminer-18/cfg.yml' # for  Aminer-18
#cfg_path='../config/WhoIsWho-SND/cfg_50.yml' # for WhoisWho-SND

python process/01_DF.py --run_model='run' --cfg_path=${cfg_path}
python process/02_BERT_MLM_TASK.py --run_model='run' --cfg_path=${cfg_path} # Require GPU
python process/03_IDF_W2V.py --run_model='run' --cfg_path=${cfg_path}
python process/04_ADJ_N2V.py --run_model='run' --cfg_path=${cfg_path}
python process/05_EMBLayer.py --run_model='run' --cfg_path=${cfg_path}

