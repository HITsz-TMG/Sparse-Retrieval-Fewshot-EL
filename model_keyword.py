
from transformers.models.electra import ElectraTokenizer,ElectraModel,ElectraConfig
from transformers import  BertTokenizer, BertConfig, BertModel

from torch import nn
import torch

class KeyWordModel(nn.Module):
    def __init__(self,pretrained_model,keyword_num,device):
        super(KeyWordModel, self).__init__()

        self.config = ElectraConfig.from_pretrained(pretrained_model)
        self.model = ElectraModel(self.config).from_pretrained(pretrained_model)

        self.model.resize_token_embeddings(self.config.vocab_size+20)
        self.net = nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),
        )
        self.keyword_num = keyword_num
        self.device = device
        self.threshold = 0.5

    def forward(self,input_ids,attention_mask,labels = None):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        logits = self.net(last_hidden_state).squeeze(-1)

        if self.training:
            logits_mask = attention_mask > 0
            logits = logits[logits_mask]
            labels = labels[logits_mask]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
            return loss
        else:
            max_logits_indices = logits.sort(-1,True).indices
            sorted_ids = input_ids.gather(1,max_logits_indices)

            return sorted_ids





