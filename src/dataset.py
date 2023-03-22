import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.kobert_tokenizer import KoBertTokenizer


class KEMDyDataset(Dataset):
    def __init__(self, modal='text', k=1, kind='train', l_type='valence'):
        self.modal = modal
        self.k = k
        self.kind = kind
        self.l_type = l_type
        self.category_dict = {'neutral': 0, 'happy': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'disqust': 5, 'fear': 6}

        self.path = './data/KEMDy20_v1_1/wav'
        self.label_path = './data/KEMDy20_v1_1/annotation_e'

        self.data = None
        self.mask = None
        self.label = None

        self.sess_seg = []

        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        # Session Validation
        if kind == 'train':
            index = list(range(1, 41))
            del index[8*(k-1):8*k]
            self.load_data(index)
        else:
            index = list(range(8*(k-1)+1, 8*k+1))
            self.load_data(index)

    def load_data(self, index):
        sentence = []
        labels = []

        for session_file in os.listdir(self.label_path):
            session = session_file.split('_')[1]
            session_number = int(session[4:])
            if session_number in index:
                label_dataframe = pd.read_csv(f'{self.label_path}/{session_file}', index_col=0, header=0)
                session = session[:4] + 'ion' + session[4:]
                for i, row in label_dataframe.iterrows():
                    self.sess_seg.append((session_number, i))
                    segment = row['Segment ID']

                    if self.l_type == 'valence':
                        valence = round(float(row['Valence']))
                        labels.append(valence)
                    elif self.l_type == 'arousal':
                        arousal = round(float(row['Arousal']))
                        labels.append(arousal)
                    elif self.l_type == 'emotion':
                        emotion = row['Emotion']
                        labels.append(self.category_dict.get(emotion))

                    # Load Text
                    if self.modal == 'text':
                        with open(f'{self.path}/{session}/{segment}.txt', 'r', encoding='cp949') as file:
                            sentence.append(file.readline().strip())

                    # Load Audio ...

        self.label = torch.tensor(labels)

        if self.modal == 'text':
            self.data, self.mask = self.tokenize(sentence)
            print(self.data.shape, self.mask.shape, self.label.shape)

    def tokenize(self, sentence):
        encode = self.tokenizer.batch_encode_plus(sentence, truncation=True, padding='max_length')

        return torch.tensor(encode['input_ids']), torch.tensor(encode['attention_mask'])

    def __getitem__(self, index):
        if self.modal == 'text':
            data, mask, label = self.data[index], self.mask[index], self.label[index]
            return data, mask, label

    def __len__(self):
        return self.data.size(0)
