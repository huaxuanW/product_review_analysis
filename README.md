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
如果其中正样本或负样本数量太少，可能无法正常`train_test_split`
```bash
# 在src/config.py里设置参数
sh train.sh
```

## Predict Product Review
---
```bash
sh inference.py
```

## Folder Tree
---
目录结构和说明
```text
DIR
│   inference.sh
│   init.sh
│   README.md
│   requirements.txt
│   train.sh
│
├───data                      #存放数据
├───outputs                   #存放预测结果
└───src                       #程序
    │   aspects_opinions.py   #提取关键词和其观点
    │   config.py             #参数
    │   RUN_THIS.py           #整个预测程序，使用训练好的模型
    │   sentiment_analysis.py #预测关键词的观点
    │   tfidf.py              #统计用户评论，得到关键词库
    │   train.py              #训练情感分类模型
    │   train_preprocess.py   #情感分类模型数据预处理
    │
    ├───external_data         #外部数据
    │       DoN               #电商词典-负面
    │       DoP               #电商词典-正面
    │       DoUM              #电商词典-中性
    │       DoUN              #电商词典-负面
    │       DoUP              #电商词典-正面
    │       hit_stopwords.txt #哈工大停用词
    │
    ├───keywords              #存放关键词库
    └───model                 #存放模型结构和训练好的模型
            0.92_max_len_200_saved_weights5.pkl #训练好的模型
            robert_model.py   #模型结构
```