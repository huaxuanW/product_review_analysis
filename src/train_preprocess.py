import pandas as pd
import re, os
from config import DATA_DIR

pos = os.path.join(DATA_DIR, "t_std_comment_his.xlsx")
neg = os.path.join(DATA_DIR, "cp_comment.xlsx")
flag = True

for path in [pos, neg]:
    df = pd.read_excel(path)[['emotion', 'comment']]
    df = df.drop_duplicates(subset=['comment'])
    def encoder(label):
        if label == '非差评':
            return 1
        else:
            return 0


    # def comment_preprocessing(text):
    #     text = re.sub(r"<br/>", "", text)
    #     text = re.sub(r"\n", "", text)
    #     text = re.sub(r"[0-9A-Za-z]", "", text)
    #     text = re.sub(r"[! @ # $ % ^ & * ( ) _ + - = ！ ¥ （ （ — — 《 》 ， 。 ？ ’ ‘ “ ” 「 」 ｜ 、 { } [ \] ~ ～ ：?]", "", text)
    #     text = re.sub(r"[🉑🏻🏼🐧👌👍👎👏💌😂😄😊😍😘🤔🥰]", "", text)
    #     return text

    df['emotion'] = df['emotion'].apply(encoder)
    # df['comment'] = df['comment'].apply(comment_preprocessing)

    if flag:
        df.to_csv(os.path.join(DATA_DIR, 'pos_comment.csv'), index = False)
        flag = False
    else:
        df.to_csv(os.path.join(DATA_DIR, 'neg_comment.csv'), index = False)

