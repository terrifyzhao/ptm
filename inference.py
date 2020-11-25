from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from model import BertForMultiTask, BertForMultiTaskWithWeight
import os
import json

GPU_NUM = 0

device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')


def read_oce_data():
    df_oce = pd.read_csv('data/OCEMOTION_a.csv', sep='\t', header=None)
    df_news = pd.read_csv('data/TNEWS_a.csv', sep='\t', header=None)
    df_oc = pd.read_csv('data/OCNLI_a.csv', sep='\t', header=None)
    data_oce = df_oce.iloc[:, 1].values
    data_news = df_news.iloc[:, 1].values
    data_oc1 = df_oc.iloc[:, 1].values
    data_oc2 = df_oc.iloc[:, 2].values

    return data_oce, data_news, [data_oc1, data_oc2]


data_oce, data_news, [data_oc1, data_oc2] = read_oce_data()

oce_dic = {'like': 0, 'happiness': 1, 'disgust': 2, 'sadness': 3, 'anger': 4, 'surprise': 5, 'fear': 6}
oce_dic = {v: k for k, v in oce_dic.items()}
news_dic = {108: 0, 102: 1, 104: 2, 107: 3, 113: 4, 116: 5, 110: 6, 115: 7, 101: 8, 109: 9, 100: 10, 103: 11, 112: 12,
            106: 13, 114: 14}
news_dic = {v: k for k, v in news_dic.items()}

oce_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=50)
oce_encodings = oce_tokenizer(data_oce.tolist(), return_tensors='pt', truncation=True, padding=True)

news_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=30)
news_encodings = news_tokenizer(data_news.tolist(), return_tensors='pt', truncation=True, padding=True)

oc_tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=40)
oc_encodings = oc_tokenizer(text=data_oc1.tolist(), text_pair=data_oc2.tolist(),
                            return_tensors='pt', truncation=True,
                            padding=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.len = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.len


oce_dataset = Dataset(oce_encodings, len(data_oce))
news_dataset = Dataset(news_encodings, len(data_news))
oc_dataset = Dataset(oc_encodings, len(data_oc1))

model = BertForMultiTaskWithWeight(num_labels1=7, num_labels2=15, num_labels3=3, device=device)
model.to(device)
model.eval()

OCE_BATCH_SIZE = 240
# OCE_BATCH_SIZE = 2
oce_loader = DataLoader(oce_dataset, batch_size=OCE_BATCH_SIZE)

NEWS_BATCH_SIZE = 400
# NEWS_BATCH_SIZE = 2
news_loader = DataLoader(news_dataset, batch_size=NEWS_BATCH_SIZE)

OC_BATCH_SIZE = 300
# OC_BATCH_SIZE = 2
oc_loader = DataLoader(oc_dataset, batch_size=OC_BATCH_SIZE)

optim = AdamW(model.parameters(), lr=5e-5)


def valid_func(valid_loader, task):
    res = []
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, task=task)

        res.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

    if task == 'oce':
        result = [oce_dic[r] for r in res]
    elif task == 'news':
        result = [news_dic[r] for r in res]
    else:
        result = res
    return result


min_valid_loss = [float('inf'), float('inf'), float('inf')]
task_name = ['oce', 'news', 'oc']
i = 0
file_name = ['ocemotion_predict.json', 'tnews_predict.json', 'ocnli_predict.json']
for loader in [oce_loader, news_loader, oc_loader]:
    print(f'************{task_name[i]} inference************')
    res = valid_func(loader, task=task_name[i])

    with open(file_name[i], 'w', encoding='utf-8')as file:
        json_dic = {}
        for j, r in enumerate(res):
            json_dic["id"] = str(j)
            json_dic["label"] = str(r)
            file.write(json.dumps(json_dic))
            file.write('\n')
        print(f'************{task_name[i]} done************')
    i += 1
