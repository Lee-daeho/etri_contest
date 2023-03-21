import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.kobert_tokenizer import KoBertTokenizer


class KEMDyDataset(Dataset):
    def __init__(self, modal='text', k=1, kind='train', l_type='valence'):
        self.modal = modal
        self.k = k
        self.kind = kind
        self.l_type = l_type

        self.path = './data/KEMDy20_v1_1/wav'
        self.label_path = './data/KEMDy20_v1_1/annotation'

        self.data = None
        self.label = None

        self.sess_seg = []

        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        # Session Validation
        if kind == 'train':
            index = list(range(1, 41))
            del index[8*(k-1):8*k]
            self.load_data(modal, index)
        else:
            index = list(range(8*(k-1)+1, 8*k+1))
            self.load_data(modal, index)

    def load_data(self, modal, index):
        sentence = []
        labels = []

        for session_file in os.listdir(self.label_path):
            session = session_file.split('_')[0]
            session_number = int(session[4:])
            if session_number in index:
                label_dataframe = pd.read_csv(f'{self.label_path}/{session_file}').iloc[1:]
                session = session[:4] + 'ion' + session[4:]
                for i, row in label_dataframe.iterrows():
                    self.sess_seg.append((session_number, i))
                    segment = row['Segment ID']

                    valence = round(float(row[' .1']))
                    arousal = round(float(row[' .2']))

                    if self.l_type == 'valence':
                        labels.append(valence)
                    elif self.l_type == 'arousal':
                        labels.append(arousal)

                    # Load Text
                    if modal == 'text':
                        with open(f'{self.path}/{session}/{segment}.txt', 'r', encoding='cp949') as file:
                            sentence.append(file.readline().strip())

                    # Load Audio ...

        self.label = torch.tensor(labels)

        if modal == 'text':
            self.data = self.tokenize(sentence)

    def tokenize(self, sentence):
        encode = self.tokenizer.batch_encode_plus(sentence, truncation=True, padding='max_length')

        return torch.tensor(encode['input_ids']), torch.tensor(encode['attention_mask'])

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]

        return data, label

    def __len__(self):
        return self.data.size(0)
