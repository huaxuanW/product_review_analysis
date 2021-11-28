# Product Review Analysis

## Business Problem
---
1. 通过模型给出电商用户评论整体的情感倾向，即非差评与差评
2. 使用无监督学习提取用户评论中相关的关键词，并通过模型预测用户对关键词的观点，即好评，差评

## Methodology
---
1. 使用`Huggingface`里预训练模型`uer/roberta-base-finetuned-jd-full-chinese`进行微调，使用最后一层pooled hidden state效果最好
2. 运用POS-Tagging，TF-IDf， TextRank的方法对用户评论进行关键词提取

## Requirements
---
```text
jieba==0.42.1
openpyxl==3.0.9
pandas
python==3.8.12
regex==2021.11.2
scikit-learn
tqdm
torch==1.10.0+cu113
transformers==4.12.3
```

## Setup Environment
---
create environment and install packages
```bash
##建立一个名为nlp的环境，并安装requirement里的包
source init.sh   # 默认安装torch + cuda 11.3
```

## Train
---
Train sentiment analysis model
需要在src/config.py 第23行修改新数据文件名
```bash
# 在src/config.py里设置参数
sh train.sh
```

## Predict Product Review
---
```bash
sh inference.py
```