from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from model2 import BertForMultiTask, BertForMultiTaskWithWeight
import os
import torch.nn as nn

GPU_NUM = 0

device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = f'{GPU_NUM}'


def read_oce_data():
    df = pd.read_csv('data/OCEMOTION_train.csv', sep='\t', header=None)
    train, valid = train_test_split(df, test_size=0.1, random_state=1000)

    train_question = train.iloc[:, 1].values
    train_label = train.iloc[:, 2].values

    valid_question = valid.iloc[:, 1].values
    valid_label = valid.iloc[:, 2].values
    return train_question, train_label, valid_question, valid_label


def read_news_data():
    df = pd.read_csv('data/TNEWS_train.csv', sep='\t', header=None)
    train, valid = train_test_split(df, test_size=0.1, random_state=1000)

    train_question = train.iloc[:, 1].values
    train_label = train.iloc[:, 2].values

    valid_question = valid.iloc[:, 1].values
    valid_label = valid.iloc[:, 2].values
    return train_question, train_label, valid_question, valid_label


def read_oc_data():
    df = pd.read_csv('data/OCNLI_train.csv', sep='\t', header=None)
    train, valid = train_test_split(df, test_size=0.1, random_state=1000)

    train_question1 = train.iloc[:, 1].values
    train_question2 = train.iloc[:, 2].values
    train_label = train.iloc[:, 3].values

    valid_question1 = valid.iloc[:, 1].values
    valid_question2 = valid.iloc[:, 2].values
    valid_label = valid.iloc[:, 3].values
    return train_question1, train_question2, train_label, valid_question1, valid_question2, valid_label


oce_train_question, oce_train_label, oce_valid_question, oce_valid_label = read_oce_data()
news_train_question, news_train_label, news_valid_question, news_valid_label = read_news_data()
oc_train_question1, oc_train_question2, oc_train_label, oc_valid_question1, oc_valid_question2, oc_valid_label = read_oc_data()

# sentence_len = []
# for q1, q2 in zip(oc_train_question1, oc_train_question2):
#     l = len(q1) + len(q2)
#     sentence_len.append(l)
# print(np.mean(sentence_len))
# print(np.percentile(sentence_len, 80))
# print(np.percentile(sentence_len, 90))

oce_dic = {'like': 0, 'happiness': 1, 'disgust': 2, 'sadness': 3, 'anger': 4, 'surprise': 5, 'fear': 6}
news_dic = {108: 0, 102: 1, 104: 2, 107: 3, 113: 4, 116: 5, 110: 6, 115: 7, 101: 8, 109: 9, 100: 10, 103: 11, 112: 12,
            106: 13, 114: 14}

oce_train_label = [oce_dic[l] for l in oce_train_label]
oce_valid_label = [oce_dic[l] for l in oce_valid_label]

news_train_label = [news_dic[l] for l in news_train_label]
news_valid_label = [news_dic[l] for l in news_valid_label]

oce_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=80)
oce_train_encodings = oce_tokenizer(oce_train_question.tolist(), return_tensors='pt', truncation=True, padding=True)
oce_valid_encodings = oce_tokenizer(oce_valid_question.tolist(), return_tensors='pt', truncation=True, padding=True)

news_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=30)
news_train_encodings = news_tokenizer(news_train_question.tolist(), return_tensors='pt', truncation=True, padding=True)
news_valid_encodings = news_tokenizer(news_valid_question.tolist(), return_tensors='pt', truncation=True, padding=True)

oc_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=50)
oc_train_encodings = oc_tokenizer(text=oc_train_question1.tolist(), text_pair=oc_train_question2.tolist(),
                                  return_tensors='pt', truncation=True,
                                  padding=True)
oc_valid_encodings = oc_tokenizer(text=oc_valid_question1.tolist(), text_pair=oc_valid_question2.tolist(),
                                  return_tensors='pt', truncation=True,
                                  padding=True)


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

news_train_dataset = Dataset(news_train_encodings, news_train_label)
news_valid_dataset = Dataset(news_valid_encodings, news_valid_label)

oc_train_dataset = Dataset(oc_train_encodings, oc_train_label)
oc_valid_dataset = Dataset(oc_valid_encodings, oc_valid_label)

if os.path.exists('bert.p'):
    print('************load model************')
    model = BertForMultiTaskWithWeight(num_labels1=7, num_labels2=15, num_labels3=3, device=device)
    model.to(device)
else:
    model = BertForMultiTask.from_pretrained('./bert', num_labels1=7, num_labels2=15, num_labels3=3)
    model.to(device)
model.train()

OCE_BATCH_SIZE = 130
# OCE_BATCH_SIZE = 2
oce_train_loader = DataLoader(oce_train_dataset, batch_size=OCE_BATCH_SIZE)
oce_valid_loader = DataLoader(oce_valid_dataset, batch_size=OCE_BATCH_SIZE)

NEWS_BATCH_SIZE = 200
# NEWS_BATCH_SIZE = 2
news_train_loader = DataLoader(news_train_dataset, batch_size=NEWS_BATCH_SIZE)
news_valid_loader = DataLoader(news_valid_dataset, batch_size=NEWS_BATCH_SIZE)

OC_BATCH_SIZE = 200
# OC_BATCH_SIZE = 2
oc_train_loader = DataLoader(oc_train_dataset, batch_size=OC_BATCH_SIZE)
oc_valid_loader = DataLoader(oc_valid_dataset, batch_size=OC_BATCH_SIZE)

optim = AdamW(model.parameters(), lr=5e-5)
loss_fct = nn.CrossEntropyLoss()


def train_func(task):
    train_loss = 0
    train_f1 = 0
    pbar = tqdm(oce_train_loader, news_train_loader, oc_train_loader)
    for batches in pbar:
        for batch in batches:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, outputs, bert, fc = model(input_ids, attention_mask=attention_mask, labels=labels, task=task)
            train_loss += loss.item()
            loss.backward()
            optim.step()
            f1 = f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
            train_f1 += f1

        pbar.update()
        pbar.set_description(f'train loss:{loss.item()}, train f1:{f1}')

    return train_loss / len(train_loader), train_f1 / len(train_loader), bert, fc

    # torch.save(bert, 'bert.p')


def valid_func(valid_loader, task):
    valid_loss = 0
    valid_f1 = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, outputs, _, _ = model(input_ids, attention_mask=attention_mask, labels=labels, task=task)
            valid_loss += loss.item()
            valid_f1 += f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')

    return valid_loss / len(valid_loader), valid_f1 / len(valid_loader)


min_valid_loss = [float('inf'), float('inf'), float('inf')]
task_name = ['oce', 'news', 'oc']
# task_name = ['news','oce',  'oc']
for epoch in range(100):
    print(f'************epoch {epoch + 1}************')
    i = 0
    for train_loader, valid_loader in zip([oce_train_loader, news_train_loader, oc_train_loader],
                                          [oce_valid_loader, news_valid_loader, oc_valid_loader]):
        print(f'{task_name[i]} train')
        train_loss, train_f1, bert, fc = train_func(train_loader, task=task_name[i])
        print(f'{task_name[i]} train loss: {train_loss:.4f}, train f1: {train_f1:.4f}')

        print(f'{task_name[i]} valid')
        valid_loss, valid_f1 = valid_func(valid_loader, task=task_name[i])
        print(f'{task_name[i]} valid loss: {valid_loss:.4f}, valid f1: {valid_f1:.4f}')

        if min_valid_loss[i] > valid_loss:
            min_valid_loss[i] = valid_loss
            torch.save(fc, f'fc{i + 1}.p')
            print('save fc layer done')
            torch.save(bert, f'bert.p')
            print('save bert done')

        i += 1
