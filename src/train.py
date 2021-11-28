import torch, os
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from transformers import AdamW, BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from config import MODEL_NAME, MAX_LEN, DATA_DIR, CODE_DIR
from model.robert_model import RobertClassifier

BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 5e-6  
EPOCHS = 10



robert = BertModel.from_pretrained(MODEL_NAME)

pos = pd.read_csv(os.path.join(DATA_DIR, 'pos_comment.csv'))
pos = pos.drop_duplicates().dropna()
neg = pd.read_csv(os.path.join(DATA_DIR, 'neg_comment.csv'))
neg = neg.drop_duplicates().dropna()
neg = neg.sample(n=len(pos), random_state= 2021, replace= False)
df = pd.concat([pos, neg], axis=0)
df = df.dropna()


X_train, X_valid, y_train,  y_valid = train_test_split(df['comment'], df['emotion'], random_state=2021, test_size=0.2, stratify=df['emotion'])

print(y_train.value_counts())

print(y_valid.value_counts())


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

X_train = tokenizer.batch_encode_plus(
    X_train.tolist(),
    max_length= MAX_LEN,
    padding= True,
    truncation= True
)

X_valid = tokenizer.batch_encode_plus(
    X_valid.tolist(),
    max_length= MAX_LEN,
    padding= True,
    truncation= True
)

train_seq = torch.tensor(X_train['input_ids'])
train_mask = torch.tensor(X_train['attention_mask'])
train_token_type = torch.tensor(X_train['token_type_ids'])
y_train = torch.tensor(y_train.tolist())


valid_seq = torch.tensor(X_valid['input_ids'])
valid_mask = torch.tensor(X_valid['attention_mask'])
valid_token_type = torch.tensor(X_valid['token_type_ids'])
y_valid = torch.tensor(y_valid.tolist())

train_data = TensorDataset(train_seq, train_mask, train_token_type, y_train)
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

valid_data = TensorDataset(valid_seq, valid_mask, valid_token_type, y_valid)
valid_sampler = SequentialSampler(valid_data)
valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

model = RobertClassifier(robert)

model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr= LR)
criterion = nn.CrossEntropyLoss()


def train(epoch):

    model.train()

    total_acc = 0
    total_loss = 0

    train_bar = tqdm(train_data_loader)

    for batch in train_bar:

        train_bar.set_description('Epoch %i train' % epoch)

        batch = [r.to(DEVICE) for r in batch]

        sent_id, mask, types, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()    

        # get model predictions for the current batch
        preds = model(sent_id, mask, types).squeeze(1)

        # compute the loss between actual and predicted values
        loss = criterion(preds, labels)

        # add on to the total loss
        total_loss += loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # update parameters
        optimizer.step()

        preds = torch.argmax(preds, dim=1)
        acc = torch.sum(preds == labels).item() / len(preds)
        total_acc += acc
    
        train_bar.set_postfix(loss=loss.item(), acc=acc)

    average_loss = total_loss / len(train_data_loader)
    average_acc = total_acc / len(train_data_loader)
    print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

    return


def evaluate(epoch):

    # deactivate dropout layers
    model.eval()

    total_acc = 0
    total_loss = 0

    valid_bar = tqdm(valid_data_loader)

    for batch in valid_bar:
        valid_bar.set_description('Epoch %i valid' % epoch)

        batch = [t.to(DEVICE) for t in batch]

        sent_id, mask, types, labels = batch

        # deactivate autograd
        with torch.no_grad():
            
            # model predictions
            preds = model(sent_id, mask, types)

            # compute the validation loss between actual and predicted values
            loss = criterion(preds, labels)

            total_loss += loss.item()

            preds = torch.argmax(preds, dim=1)
            acc = torch.sum(preds == labels).item() / len(labels)
            total_acc += acc

        valid_bar.set_postfix(loss=loss.item(), acc=acc)

    average_loss = total_loss / len(valid_data_loader)
    average_acc = total_acc / len(valid_data_loader)

    print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

    return average_acc


best_acc = 0
for epoch in range(1, EPOCHS + 1):

    train(epoch)

    valid_acc = evaluate(epoch)

    print('\n')

    if valid_acc > best_acc:

        best_acc = valid_acc

        torch.save(model.state_dict(), os.path.join(CODE_DIR, 'model/saved_weights5.pkl'))
