import torch, os, glob, logging, re, sys

import pandas as pd

from transformers import BertTokenizer, BertModel

from model.robert_model import RobertClassifier

from config import MAX_LEN, DATA_DIR, MODEL_NAME, MODEL_PATH

logging.disable(logging.WARNING)


def read_data(data_type):

    files = glob.glob(os.path.join(DATA_DIR, f'*.{data_type}'))
    if not files:
        print('Does not find data')
    if len(files) > 1:
        
        sys.exit(f'Found more than one {data_type} type of data, please just use one dataset')

    if data_type == 'csv':
        Data = pd.read_csv(files[0])
    
    elif data_type == 'xlsx':
        Data = pd.read_excel(files[0])
    
    return Data['comment'].tolist(), Data['emotion'].tolist()

def load_model(device, model_path, model_name):
    
    tokenizer = BertTokenizer.from_pretrained(model_name)

    robert = BertModel.from_pretrained(model_name)

    model = RobertClassifier(robert).to(device)

    model.load_state_dict(torch.load(model_path))

    return model, tokenizer

def comment_preprocessing(text):
    text = re.sub(r"<br/>", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[0-9A-Za-z]", "", text)
    text = re.sub(r"[! @ # $ % ^ & * ( ) _ + - = ！ ¥ （ （ — — 《 》 ， 。 ？ ’ ‘ “ ” 「 」 ｜ 、 { } [ \] ~ ～ ：?]", "", text)
    text = re.sub(r"[🉑🏻🏼🐧👌👍👎👏💌😂😄😊😍😘🤔🥰]", "", text)
    return text



def get_opinion(comment, model, tokenizer, device):

    labels = ['差评', '非差评']

    comment = comment_preprocessing(comment)

    token = tokenizer.batch_encode_plus([comment], max_length= MAX_LEN, padding= True, truncation= True)

    input_ids = torch.tensor(token['input_ids']).to(device)
    attention_mask = torch.tensor(token['attention_mask']).to(device)
    token_type_ids = torch.tensor(token['token_type_ids']).to(device)
    with torch.no_grad():
        preds = model(input_ids, attention_mask, token_type_ids)
    preds = preds.detach().cpu().numpy().argmax()
    return labels[preds]


if __name__ == '__main__':
    comments, labels  = read_data('xlsx')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = load_model(DEVICE, MODEL_PATH, MODEL_NAME)
    
    for i, c in enumerate(comments):
        print(c)
        print(get_opinion(c, model, tokenizer, DEVICE))
        print('*' * 10)
        if i > 10:
            break