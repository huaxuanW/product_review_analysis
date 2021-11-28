import jieba, re, torch

import jieba.analyse

from sentiment_analysis import read_data, load_model, get_opinion

from config import MODEL_NAME, MODEL_PATH


def read_text_data(path):
    res = []
    with open(path, 'r', encoding='utf-8' ) as f:
        for line in f.readlines():
            res.append(line.strip())
    return res


def get_primary_dim(comment, keyword):
    res = []
    
    candidate1 = set(jieba.analyse.extract_tags(comment, topK=10,  allowPOS=('ns', 'n', 'vn' ,'v', 'l')))
    candidate2 = set(jieba.analyse.textrank(comment, topK=10,  allowPOS=('ns', 'n', 'vn' ,'v', 'l')))
    candidate = list(candidate1 | candidate2)
    for c in candidate:
        if c in keyword:
            res.append(c)
    if len(res) < 1:
        return ['其他']
    return res


def get_pairs(chunks, primary_dims):
    res = {}
    for w in primary_dims:
        for chunk in chunks:
            if w in chunk:
                res[w] = chunk
                break
    return res


def get_chunk(comment):
    comment = re.sub(r"\n", ",", comment)
    comment = re.sub(r"<br/>", ",", comment)
    comment_chunks = re.sub(r"[! ！  ， 。 ？  ； 、 ,]", " ", comment).split()
    return comment_chunks


def aspect_opinion(comment, model, tokenizer, device, keyword):
    res = {}
    
    primary_dims = get_primary_dim(comment, keyword)

    chunks = get_chunk(comment)

    pairs = get_pairs(chunks, primary_dims)

    for key, value in pairs.items():
        opinion = get_opinion(value, model, tokenizer, device)
        res[key] = opinion
    
    return res


if __name__ == '__main__':
    from tfidf import get_key_aspects

    comments, labels  = read_data('xlsx')

    get_key_aspects(comments)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = load_model(DEVICE, MODEL_PATH, MODEL_NAME)
    
    for i, c in enumerate(comments):
        print(aspect_opinion(c, model, tokenizer, DEVICE))
        if i > 10:
            break