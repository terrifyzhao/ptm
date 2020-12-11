from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import BertForMultiTask
from torch.optim import AdamW
# from transformers import AdamW
import os

GPU_NUM = 0

device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = f'{GPU_NUM}'

BATCH_SIZE = 450
# BATCH_SIZE = 40
SIZE = 10000000


def read_oce_data():
    df = pd.read_csv('data/OCEMOTION_train1128.csv', sep='\t')[0:SIZE]
    train, valid = train_test_split(df, test_size=0.05, random_state=1000)

    train_question = train.iloc[:, 1].values
    train_label = train.iloc[:, 2].values

    valid_question = valid.iloc[:, 1].values
    valid_label = valid.iloc[:, 2].values
    return train_question, train_label, valid_question, valid_label


oce_train_question, oce_train_label, oce_valid_question, oce_valid_label = read_oce_data()

dic = {}
for v in oce_train_label:
    dic[v] = dic.get(v, 0) + 1
print(dic)
label_dic = {}
for k, v in dic.items():
    label_dic[k] = len(label_dic)
print(label_dic)

oce_train_label = [label_dic[l] for l in oce_train_label]
oce_valid_label = [label_dic[l] for l in oce_valid_label]

data = oce_train_question.tolist()
data.append(oce_valid_question.tolist())
sentence_len = [len(s) for s in data]
print(np.mean(sentence_len))
print(np.percentile(sentence_len, 80))
print(np.percentile(sentence_len, 90))

tokenizer = BertTokenizer.from_pretrained('./bert', model_max_length=80)
train_encodings = tokenizer(oce_train_question.tolist(), return_tensors='pt', truncation=True, padding=True)
valid_encodings = tokenizer(oce_valid_question.tolist(), return_tensors='pt', truncation=True, padding=True)


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


train_dataset = Dataset(train_encodings, oce_train_label)
valid_dataset = Dataset(valid_encodings, oce_valid_label)

if os.path.exists('best_model.p'):
    print('************load model************')
    model = torch.load('best_model.p')
else:
    model = BertForMultiTask.from_pretrained('./bert', num_labels1=7, num_labels2=15, num_labels3=3)
    model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# optim = AdamW(model.parameters(), lr=5e-5)
lr = 5e-5
bert_param = []
encoder_id = []
lr_eps = 0.95
for i in range(11, -1, -1):
    params_id = list(map(id, model.bert.encoder.layer[i].parameters()))
    encoder_id.extend(params_id)
    params = model.bert.encoder.layer[i].parameters()
    bert_param.append({'params': params, 'lr': lr})
    lr = lr * lr_eps

encoder_id.extend(list(map(id, model.inter.parameters())))
encoder_id.extend(list(map(id, model.classifier1.parameters())))
encoder_id.extend(list(map(id, model.classifier2.parameters())))
encoder_id.extend(list(map(id, model.classifier3.parameters())))

base_param = filter(lambda p: id(p) not in encoder_id, model.parameters())
bert_param.append({'params': base_param, 'lr': 5e-5})
bert_param.append({'params': model.inter.parameters(), 'lr': 1e-4})
bert_param.append({'params': model.classifier1.parameters(), 'lr': 1e-4})
bert_param.append({'params': model.classifier2.parameters(), 'lr': 1e-4})
bert_param.append({'params': model.classifier3.parameters(), 'lr': 1e-4})
optim = AdamW(bert_param)

loss_fc = torch.nn.CrossEntropyLoss()

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

min_valid_loss = float('inf')


def train_func():
    global min_valid_loss
    train_loss = 0
    train_f1 = 0
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs, _, _ = model(input_ids, attention_mask=attention_mask, labels=labels, task='oce')
        loss = loss_fc(outputs.view(-1, 7), labels.view(-1))
        train_loss += loss.item()
        loss.backward()
        optim.step()
        f1 = f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')
        train_f1 += f1

        pbar.update()
        pbar.set_description(f'train loss:{loss.item()}, train f1:{f1}')

        if (step != 0 and (step + 1) % 25 == 0) or step == len(pbar) - 1:
            valid_loss, valid_f1 = test_func()
            print(f'valid loss: {valid_loss:.4f}, valid f1: {valid_f1:.4f}')

            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(model, 'best_model.p')
                print('save model done')

    return train_loss / len(train_loader), train_f1 / len(train_loader)


def test_func():
    valid_loss = 0
    valid_f1 = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs, _, _ = model(input_ids, attention_mask=attention_mask, labels=labels, task='oce')
            loss = loss_fc(outputs.view(-1, 7), labels.view(-1))
            valid_loss += loss.item()
            valid_f1 += f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='macro')

    return valid_loss / len(valid_loader), valid_f1 / len(valid_loader)


for epoch in range(100):
    print(f'************epoch {epoch}************')
    print('start train')
    train_loss, train_f1 = train_func()
    print(f'train loss: {train_loss:.4f}, train f1: {train_f1:.4f}')
    # scheduler.step()
    print('start valid')

    print('')
