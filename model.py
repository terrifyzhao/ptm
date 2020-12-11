from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.file_utils import *
import torch.nn.functional as F
import torch


class BertForMultiTaskWithWeight(nn.Module):
    def __init__(self, num_labels1, num_labels2, num_labels3, device):
        super().__init__()
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3

        model = torch.load('bert.p', map_location=device)
        self.bert = model[0]
        self.classifier1 = model[1]
        self.classifier2 = model[2]
        self.classifier3 = model[3]

        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            task=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)

        if labels is None:
            if task == 'oce':
                return logits1
            elif task == 'news':
                return logits2
            elif task == 'oc':
                return logits3

        if task == 'oce':
            return logits1, self.bert, self.classifier1
        elif task == 'news':
            return logits2, self.bert, self.classifier2
        elif task == 'oc':
            return logits3, self.bert, self.classifier3


class InterpretationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        # [bs, seq_len]
        out = self.dense(x)
        alpha = F.softmax(out, dim=1)
        # [bs, seq_len,768] [bs, seq]
        out = x * alpha
        out = torch.sum(out, dim=1)

        # out = sum(out, axis=1)
        return out


class BertForMultiTask(BertPreTrainedModel):
    def __init__(self, config, num_labels1, num_labels2, num_labels3):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_dropout = nn.Dropout(0.1)

        self.inter = InterpretationLayer(config)

        self.classifier1 = nn.Linear(config.hidden_size, num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, num_labels3)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            task=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # pooled_output = outputs[1]

        out = self.inter(outputs[0])

        pooled_output = self.dropout(out)
        logits1 = self.classifier1(pooled_output)
        logits1 = self.cls_dropout(logits1)
        logits2 = self.classifier2(pooled_output)
        logits2 = self.cls_dropout(logits2)
        logits3 = self.classifier3(pooled_output)
        logits3 = self.cls_dropout(logits3)

        if task == 'oce':
            return logits1, self.bert, self.classifier1
        elif task == 'news':
            return logits2, self.bert, self.classifier2
        elif task == 'oc':
            return logits3, self.bert, self.classifier3


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=5, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
