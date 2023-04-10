import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.kobert_tokenizer import KoBertTokenizer
from transformers import AutoTokenizer
import torchaudio
from transformers import Wav2Vec2Processor,Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import vocab
import json
from tqdm import tqdm 
import random

import re


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

        if modal=='text':
            self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
            # self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
        if modal=='wav':    
            model_name_or_path = "kresnik/wav2vec2-large-xlsr-korean"
            # self.wav_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name_or_path)
            # self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
            # self.processor = Wav2Vec2Processor(feature_extractor=self.wav_feature_extractor, tokenizer=self.wav_tokenizer)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
            self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
            print(f"The target sampling rate: {self.target_sampling_rate}")

        # # Session Validation
        # if kind == 'train':
        #     index = list(range(1, 41))
        #     del index[8*(k-1):8*k]
        #     self.load_data(index)
        # else:
        #     index = list(range(8*(k-1)+1, 8*k+1))
        #     self.load_data(index)

        self.max_len = 512
        self.json_path = f'./data/KEMDy_20_v1_1_json/{kind}_data_fold{k}.json'
        dialogues = self.load_kemdy_turn()
        self.build_dataset(dialogues)



    def load_kemdy_turn(self):

        # train = if self.kind == 'train'
        emotion_vocab = vocab.Vocab.from_dict(torch.load('results/vocabs/emotion_vocab.pkl'))
        data = json.load(open(self.json_path, 'r'), encoding='utf8')
        
        speaker_pools = json.load(open('./data/KEMDy_20_v1_1_json/name_pool_ko', 'r'))
        dialogues = []
        for dialog in tqdm(data,
                desc='processing file {}'.format(self.json_path)):
            dialogue = []
            t_vocab = vocab.Vocab()
            speaker_vocab = vocab.Vocab()
            for utterance in dialog:
                speaker = utterance.get('speaker').upper()
                text = utterance.get('text') 
                emotion = utterance.get('label')
                
                speaker = speaker_pools[t_vocab.word2index(speaker, train=True)]
                speaker_vocab.word2index(speaker, train=True)
                turn_data = {}
                turn_data['speaker'] = speaker
                turn_data['text'] = text
                if emotion is not None:
                    emotion_idx = emotion_vocab.word2index(emotion)
                else:
                    emotion_idx = -1
                turn_data['label'] = emotion_idx
                dialogue.append(turn_data)
            dialogues.append(dialogue)
            # print(speaker_vocab)
        # speaker_vocab = speaker_vocab.prune_by_count(30)
        i=0
        for speaker_name in speaker_vocab.counts.keys():
            self.tokenizer.add_tokens(speaker_name)
            i+=1
        print("num of new tokens: ", i)
        return dialogues


    def build_dataset(self, dialogues):
        ret_utterances = []
        ret_labels = []
        for dialogue in dialogues:
            

            utterance_ids = []
            query = 'For utterance:'
            query_ids = self.tokenizer(query)['input_ids'][1:-1]
            for idx, turn_data in enumerate(dialogue):
                # clean_text = re.sub('[a-zA-Z]/|\*|\+|\/', '', turn_data['text']).replace('  ', ' ').lstrip(' ').rstrip(' ')
                # text_with_speaker = turn_data['speaker'] + ':' + clean_text
                text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
                token_ids = self.tokenizer(text_with_speaker)['input_ids'][1:]
                utterance_ids.append(token_ids)
                if turn_data['label'] < 0:
                    continue
                full_context = [2]      # [CLS]
                lidx = 0
                for lidx in range(idx):
                    total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 2
                    if total_len + len(utterance_ids[idx]) <= self.max_len:
                        break
                lidx = max(lidx, idx-2)
                for item in utterance_ids[lidx:]:
                    full_context.extend(item)


                query_idx = idx
                prompt = dialogue[query_idx]['speaker'] + ' feels [MASK]'
                full_query = query_ids + utterance_ids[query_idx][:-1] + self.tokenizer(prompt)['input_ids'][1:]
                # full_query = utterance_ids[query_idx]
                input_ids = full_context + full_query
                input_ids = self.pad_to_len(input_ids, self.max_len, 1)
                ret_utterances.append(input_ids)               
                ret_labels.append(dialogue[query_idx]['label'])


                # these codes make samples of random query with sharing same context 
                # if self.kind == 'train' and idx > 3 and torch.rand(1).item() < 0.2:
                #     query_idx = random.randint(lidx, idx-1)
                #     if dialogue[query_idx]['label'] < 0:
                #         continue
                #     prompt = dialogue[query_idx]['speaker'] + ' [MASK]'
                #     full_query = query_ids + utterance_ids[query_idx] + self.tokenizer(prompt)['input_ids'][1:]
                #     # full_query = utterance_ids[query_idx]
                #     input_ids = full_context + full_query
                #     input_ids = self.pad_to_len(input_ids, self.max_len, 1)
                #     ret_utterances.append(input_ids)
                #     ret_labels.append(dialogue[query_idx]['label'])
        
        # print some samples 
        if self.kind =='train':
            for i in range(5):
                print(ret_utterances[i])
                print(self.tokenizer.decode(ret_utterances[i]))
            
        ret_masks = 1- np.array([np.array(u)==1 for u in ret_utterances])

        self.mask = torch.tensor(ret_masks)
        self.data = torch.tensor(ret_utterances)
        self.label = torch.tensor(ret_labels)
        
    def pad_to_len(self, list_data, max_len, pad_value):
        list_data = list_data[-max_len:]
        len_to_pad = max_len - len(list_data)
        pads = [pad_value] * len_to_pad
        list_data.extend(pads)
        return list_data
    
    
    def load_data(self, index):
        sound = []
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
                    if self.modal == 'wav':
                        sound.append(self.speech_file_to_array_fn(f'{self.path}/{session}/{segment}.wav'))
                    

        self.label = torch.tensor(labels)

        if self.modal == 'text':
            self.data, self.mask = self.tokenize(sentence)
            print(self.data.shape, self.mask.shape, self.label.shape)

        if self.modal == 'wav':
            # results = self.wav_feature_extractor(sound, sampling_rate=self.target_sampling_rate, padding='max_length', return_attention_mask=True)
            results = self.processor(sound, sampling_rate=self.target_sampling_rate, padding="max_length", max_length=45520, truncation=True)
            self.data = torch.tensor(results['input_values'])
            self.mask = torch.tensor(results['attention_mask'])
            print(self.data.shape, self.mask.shape, self.label.shape)
            


    def tokenize(self, sentence):
        encode = self.tokenizer.batch_encode_plus(sentence, truncation=True, padding='max_length')

        return torch.tensor(encode['input_ids']), torch.tensor(encode['attention_mask'])

    
    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech



    def __getitem__(self, index):
        if self.modal == 'text':
            data, mask, label = self.data[index], self.mask[index], self.label[index]
            return data, mask, label
            # data, label = self.data[index], self.label[index]
            # return data, label
        if self.modal == 'wav':
            data, mask, label = self.data[index], self.mask[index], self.label[index]
            return data, mask, label

    def __len__(self):
        # return self.data.size(0)
        return len(self.data[0])
