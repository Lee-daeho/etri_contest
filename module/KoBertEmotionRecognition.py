import torch
import torch.nn as nn
from transformers import BertModel

import time


class KoBERTEmotionRecognition(nn.Module):
    def __init__(self):
        super(KoBERTEmotionRecognition, self).__init__()

        self.encoder = BertModel.from_pretrained("monologg/kobert")
        self.mlp = nn.Linear(in_features=768, out_features=7)

    def forward(self, ids, mask):
        encode_out = self.encoder(ids, mask)
        out = self.mlp(encode_out.pooler_output)

        return out

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'KoBert_Classifier'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
