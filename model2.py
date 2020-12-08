from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.file_utils import *
import torch.nn.functional as F


class BertForMultiTaskWithWeight(nn.Module):
    def __init__(self, num_labels1, num_labels2, num_labels3, device):
        super().__init__()
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3

        self.bert = torch.load('bert.p', map_location=device)
        self.classifier1 = torch.load('fc1.p', map_location=device)
        self.classifier2 = torch.load('fc2.p', map_location=device)
        self.classifier3 = torch.load('fc3.p', map_location=device)

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

        loss_fct = nn.CrossEntropyLoss()

        if task == 'oce':
            loss = loss_fct(logits1.view(-1, self.num_labels1), labels.view(-1))
            return loss, logits1, self.bert, self.classifier1
        elif task == 'news':
            loss = loss_fct(logits2.view(-1, self.num_labels2), labels.view(-1))
            return loss, logits2, self.bert, self.classifier2
        elif task == 'oc':
            loss = loss_fct(logits3.view(-1, self.num_labels3), labels.view(-1))
            return loss, logits3, self.bert, self.classifier3


class BertForMultiTask(BertPreTrainedModel):
    def __init__(self, config, num_labels1, num_labels2, num_labels3):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)

        # return logits1, logits2, logits3, self.bert, self.classifier1, self.classifier2, self.classifier3

        if task == 'oce':
            # loss_fct = FocalLoss(logits=True)
            # loss = loss_fct(logits1.view(-1, self.num_labels1), labels.view(-1))
            return logits1, self.bert, self.classifier1
        elif task == 'news':
            # loss = loss_fct(logits2.view(-1, self.num_labels2), labels.view(-1))
            return logits2, self.bert, self.classifier2
        elif task == 'oc':
            # loss = loss_fct(logits3.view(-1, self.num_labels3), labels.view(-1))
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
