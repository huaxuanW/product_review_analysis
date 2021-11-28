import torch, os

import pandas as pd

from tfidf import get_key_aspects

from sentiment_analysis import read_data, load_model, get_opinion

from aspects_opinions import aspect_opinion

from config import DATA_TYPE, SAVED_PATH, MODEL_PATH, MODEL_NAME

from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Total Six Steps\n')

print('1.Load Data\n')
comments, labels = read_data(DATA_TYPE)


print('2.Load Model\n')
model, tokenizer = load_model(DEVICE, MODEL_PATH, MODEL_NAME)


print('3.Predict Sentiment\n')
pred_labels = []
for c in tqdm(comments):
    pred_labels.append(get_opinion(c, model, tokenizer, DEVICE))


print('4.Generate Key Aspects\n')
keyword = get_key_aspects(comments)


print('5.Extract Aspects And Opinion\n')
aspects = []
for c in tqdm(comments):
    aspects.append(aspect_opinion(c, model, tokenizer, DEVICE, keyword))


print('6.Save Data\n')
dt = pd.DataFrame(zip(comments, labels, pred_labels, aspects), columns=['comment', 'emotion', 'pred_emotion', 'aspect_opinion'])
dt.to_csv(SAVED_PATH, index= False)
print(f'Data Saved to {SAVED_PATH}\n')