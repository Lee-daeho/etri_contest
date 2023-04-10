import torch
import torch.nn as nn
from transformers import BertModel

import time


class KoBERTEmotionMaskModel(nn.Module):
    def __init__(self):
        super(KoBERTEmotionMaskModel, self).__init__()

        self.max_len = 512
        self.num_classes = 7
        self.pad_value = 1
        self.mask_value = 4         # [MASK]
        self.f_context_encoder = BertModel.from_pretrained('monologg/kobert')
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 2)      # add new tokens of two speakers
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )

        # self.g = nn.Sequential(
        #     nn.Linear(self.dim, self.dim),
        #     )


    def forward(self, sentences, mask):
        # batch_size, max_len = sentences.shape[0], sentences.shape[-1]
        # sentences = sentences.reshape(-1, max_len)
        # mask = 1 - (sentences == (self.pad_value)).long()
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
        # feature = torch.dropout(mask_outputs, 0.1, train=self.training)
        # feature = mask_outputs
        # if self.config['output_mlp']:
        #     feature = self.g(feature)

        output = self.predictor(mask_outputs)
        return output

    # def forward(self, sentences, mask):
    #     encode_out = self.f_context_encoder(sentences, mask)
    #     out = self.mlp(encode_out.pooler_output)

    #     return out




    

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
        self.load_state_dict(torch.load(path), strict=False)
