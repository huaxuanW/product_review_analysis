import re, os

import jieba.posseg as psg

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm

from config import TOP_K, CODE_DIR

def read_words(path):
    stopwords = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
        return stopwords


def comment_preprocessing(text, stopwords, tags=False):
    text = re.sub(r"<br/>", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[0-9A-Za-z]", "", text)
    text = re.sub(r"[!@#\$%\^&*\()_+-=！¥（（——《》，。？’‘“”「」｜、\{\}\[\]~～]", "", text)
    text = re.sub(r"[🉑🏻🏼🐧👌👍👎👏💌😂😄😊😍😘🤔🥰]", "", text)
    text_list = psg.cut(text)
    if tags:
        text_list = [w for w, flag in text_list if flag in tags]
    else:
        text_list = [w for w, flag in text_list]
    text_list = [w for w in text_list if w not in stopwords]
    return text_list

def bow(train, ngram):
    

    bags = CountVectorizer(ngram_range=(ngram, ngram))

    bags.fit(train)

    train = bags.transform(train)

    return bags  


def tfidf_n_gram(train, ngram):

    tfidf = TfidfVectorizer(ngram_range=(ngram, ngram))

    tfidf.fit(train)

    train = tfidf.transform(train)

    return train


def get_key_aspects(comments, filters= True):

    stopwords = read_words(os.path.join(CODE_DIR,'external_data/hit_stopwords.txt'))

    tags = ['ns', 'n', 'vn' ,'v', 'l']

    print('Cleaning Data\n')
    cleaned_comments = []

    for c in tqdm(comments):
        cleaned_comments.append(comment_preprocessing(c, stopwords, tags))

    cleaned_comments = [' '.join(c) for c in cleaned_comments]

    n = 1
    feature = bow(cleaned_comments, n)
    feature = feature.get_feature_names_out()
    score = tfidf_n_gram(cleaned_comments, n)


    sums = score.sum(axis = 0)
    data1 = []
    for col, term in enumerate(feature):
        data1.append( (term, sums[0,col]))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False)).reset_index(drop=True)


    
    
    if filters:
        path_list = ['DoN', 'DoP', 'DoUM', 'DoUN', 'DoUP']
        filter_words = []
        for path in path_list:
            path = os.path.join(CODE_DIR,f'external_data/{path}')
            filter_words.extend(read_words(path))
        new_words = []
        for term, rank in words.head(TOP_K).values:
            if term not in filter_words:
                new_words.append(term)

    KEYWORDS_PATH = os.path.join(CODE_DIR, 'keywords/keyphrase.txt')
    with open(KEYWORDS_PATH, 'w', encoding='utf-8') as f:
        for term, rank in words.head(TOP_K).values:
            f.write(f'{term}\n')

    return new_words




if __name__ == '__main__':
    from sentiment_analysis import read_data

    comments, labels = read_data('xlsx')

    get_key_aspects(comments)