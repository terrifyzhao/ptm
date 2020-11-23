from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from model import BertForMultiTask
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def read_oce_data():
    df = pd.read_csv('data/OCEMOTION_train.csv', sep='\t')
    train, valid = train_test_split(df, test_size=0.2, random_state=1000)

    train_question = train.iloc[:, 1].values
    train_label = train.iloc[:, 2].values

    valid_question = valid.iloc[:, 1].values
    valid_label = valid.iloc[:, 2].values
    return train_question, train_label, valid_question, valid_label


def read_news_data():
    df = pd.read_csv('data/TNEWS_train.csv', sep='\t')
    train, valid = train_test_split(df, test_size=0.2, random_state=1000)

    train_question = train.iloc[:, 1].values
    train_label = train.iloc[:, 2].values

    valid_question = valid.iloc[:, 1].values
    valid_label = valid.iloc[:, 2].values
    return train_question, train_label, valid_question, valid_label


oce_train_question, oce_train_label, oce_valid_question, oce_valid_label = read_oce_data()
news_train_question, news_train_label, news_valid_question, news_valid_label = read_news_data()

oce_dic = {'like': 0, 'happiness': 1, 'disgust': 2, 'sadness': 3, 'anger': 4, 'surprise': 5, 'fear': 6}
news_dic = {108: 0, 102: 1, 104: 2, 107: 3, 113: 4, 116: 5, 110: 6, 115: 7, 101: 8, 109: 9, 100: 10, 103: 11, 112: 12,
            106: 13, 114: 14}

oce_train_label = [oce_dic[l] for l in oce_train_label]
oce_valid_label = [oce_dic[l] for l in oce_valid_label]

news_train_label = [news_dic[l] for l in news_train_label]
news_valid_label = [news_dic[l] for l in news_valid_label]

oce_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=50)
oce_train_encodings = oce_tokenizer(oce_train_question.tolist(), return_tensors='pt', truncation=True, padding=True)
oce_valid_encodings = oce_tokenizer(oce_valid_question.tolist(), return_tensors='pt', truncation=True, padding=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


oce_train_dataset = Dataset(oce_train_encodings, oce_train_label)
oce_valid_dataset = Dataset(oce_valid_encodings, oce_valid_label)

if os.path.exists('best_model.p'):
    print('************load model************')
    model = torch.load('best_model.p')
else:
    model = BertForMultiTask.from_pretrained('./bert', num_labels1=7, num_labels2=15)
    model = BertForMultiTask.from_pretrained('./bert', num_labels1=7, num_labels2=15)
    model.to(device)
model.train()

OCE_BATCH_SIZE = 240
oce_train_loader = DataLoader(oce_train_dataset, batch_size=OCE_BATCH_SIZE)
oce_valid_loader = DataLoader(oce_valid_dataset, batch_size=OCE_BATCH_SIZE)

NEWS_BATCH_SIZE = 512
news_train_loader = DataLoader(oce_train_dataset, batch_size=NEWS_BATCH_SIZE)
news_valid_loader = DataLoader(oce_valid_dataset, batch_size=NEWS_BATCH_SIZE)

optim = AdamW(model.parameters(), lr=5e-5)


def train_func():
    oce_train_loss = 0
    oce_train_f1 = 0
    pbar = tqdm(oce_train_loader)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, outputs = model(input_ids, attention_mask=attention_mask, labels1=labels)
        oce_train_loss += loss.item()
        loss.backward()
        optim.step()
        f1 = f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
        oce_train_f1 += f1

        pbar.update()
        pbar.set_description(f'train loss:{loss.item()}, train f1:{f1}')

    news_train_loss = 0
    news_train_f1 = 0
    pbar = tqdm(news_train_loader)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, outputs = model(input_ids, attention_mask=attention_mask, labels2=labels)
        news_train_loss += loss.item()
        loss.backward()
        optim.step()
        f1 = f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
        news_train_f1 += f1

        pbar.update()
        pbar.set_description(f'train loss:{loss.item()}, train f1:{f1}')

    return oce_train_loss / len(oce_train_loader), oce_train_f1 / len(oce_train_loader), \
           news_train_loss / len(news_train_loader), news_train_f1 / len(news_train_loader)


def test_func():
    valid_loss = 0
    valid_f1 = 0
    for batch in tqdm(oce_valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            valid_loss += loss.item()
            valid_f1 += f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')

    return valid_loss / len(oce_valid_loader), valid_f1 / len(oce_valid_loader)


min_valid_loss = float('inf')
for epoch in range(100):
    print('************start train************')
    oce_train_loss, oce_train_f1, news_train_loss, news_train_f1 = train_func()
    # print('************start valid************')
    # valid_loss, valid_f1 = test_func()
    # print(f'valid loss: {valid_loss:.4f}, valid_f1: {valid_f1:.4f}')

    # if min_valid_loss > valid_loss:
    #     min_valid_loss = valid_loss
    #     torch.save(model, 'best_model.p')
    #     print('save model done')
