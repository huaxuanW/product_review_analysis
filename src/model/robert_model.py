import torch.nn as nn
import torch


class RobertClassifier(nn.Module):
    def __init__(self, xlnet):
        super(RobertClassifier, self).__init__()

        self.xlnet = xlnet

        self.dropout = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(768 , 2)

        # self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        hs = self.xlnet(input_ids= input_ids, attention_mask= attention_mask, token_type_ids= token_type_ids)
        
        pooled_hs = hs[1]

        x = self.dropout(pooled_hs)

        x = self.fc1(x)

        # x = self.softmax(x)

        return x