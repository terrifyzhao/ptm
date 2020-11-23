from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.file_utils import *


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
        logits3 = self.classifier2(pooled_output)

        loss_fct = nn.CrossEntropyLoss()

        if task == 'oce':
            loss = loss_fct(logits1.view(-1, self.num_labels1), labels.view(-1))
            return loss, logits1, self.bert, self.classifier1
        elif task == 'news':
            loss = loss_fct(logits2.view(-1, self.num_labels2), labels.view(-1))
            return loss, logits2, self.bert, self.classifier2
        elif task == 'oc':
            loss = loss_fct(logits2.view(-1, self.num_labels3), labels.view(-1))
            return loss, logits3, self.bert, self.classifier3
