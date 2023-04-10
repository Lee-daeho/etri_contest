import torch
import os

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

# Dataset
from src.dataset import KEMDyDataset

# Model
from module.KoBertEmotionRecognitionMask import KoBERTEmotionMaskModel
from module.KoBertEmotionRecognition import KoBERTEmotionRecognition
from module.Wav2VecEmotionRecognition import Wav2VecEmotionRecognition

from torchnet import meter
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from src.util import out_put

from transformers import AutoConfig
import transformers
import wandb
from tqdm import tqdm


def train(modal, k, l_type, epochs, lr, decay, batch_size, file_name, use_gpu=False):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    directory = file_name.split('/')[-2]
    if not os.path.exists(f'./results/{modal}/' + directory):
        os.mkdir(f'./results/{modal}/' + directory)

    train_data = KEMDyDataset(modal=modal, k=k, kind='train', l_type=l_type)
    val_data = KEMDyDataset(modal=modal, k=k, kind='val', l_type=l_type)


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model Init
    if modal == 'text':
        # model = KoBERTEmotionRecognition().to(device)
        model = KoBERTEmotionMaskModel().to(device)
    if modal == 'wav':
        model_name_or_path = "kresnik/wav2vec2-large-xlsr-korean"
        config = AutoConfig.from_pretrained(model_name_or_path,
                                            num_labels=7,
                                            label2id = {'neutral': 0, 'happy': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'disqust': 5, 'fear': 6},
                                            id2label = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'angry', 4: 'sad', 5: 'disqust', 6: 'fear'}
                                            )

        model = Wav2VecEmotionRecognition.from_pretrained(model_name_or_path, config=config).to(device)

    # Optimization Settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)


    # meters
    loss_meter = meter.AverageValueMeter()


        
    best_f1 = 0
    best_epoch = 0

    # Training
    for epoch in tqdm(range(epochs)):
        pred_label = []
        true_label = []

        loss_meter.reset()
        for ii, (data, mask, target) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            target = target.to(device)

            output = model(data, mask)
            # ccl_reps = model.gen_f_reps(data, mask)
            # output = model.predictor(ccl_reps)

            optimizer.zero_grad()
            loss = criterion(output, target)     # direct loss
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # _, pred = output.data.topk(1, dim=1)
            # pred = pred.t().squeeze()

            # pred_label.append(pred.detach().cpu())
            # true_label.append(target.detach().cpu())

        # pred_label = torch.cat(pred_label, 0)
        # true_label = torch.cat(true_label, 0)
        
        # assert(pred_label.shape == true_label.shape)        # test

        # train_f1 = f1_score(true_label, pred_label, average='weighted')
        # train_accuracy = accuracy_score(true_label, pred_label)
        # train_precision = precision_score(true_label, pred_label, average='weighted')
        # train_recall = recall_score(true_label, pred_label, average='weighted')

        out_put('Epoch: ' + 'train' + str(epoch) +
                '| train Loss: ' + str(loss_meter.value()[0]),
                # '| train F1: ' + str(train_f1) + '| train Accuracy: ' + str(train_accuracy) +
                # '| train Precision: ' + str(train_precision) + '| train Recall: ' + str(train_recall),
                file_name)
        # wandb.log({'train_loss': loss_meter.value()[0],
        #            'train F1': train_f1}, 
        #            step=epoch)
        val_loss, val_f1, val_accuracy, val_precision, val_recall = val(model, val_loader, use_gpu)

        out_put('Epoch: ' + 'val' + str(epoch) +
                '| val F1: ' + str(val_f1) + '| val Accuracy: ' + str(val_accuracy) +
                '| val Precision: ' + str(val_precision) + '| val Recall: ' + str(val_recall),
                file_name)
        
        # wandb.log({'valid_loss': val_loss,
        #            'valid F1': val_f1},
        #            step=epoch)
        
        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            model.save(f"{file_name}_best.pth")

    model.save(f'{file_name}.pth')

    perf = f"best accuracy is {best_f1} in epoch {best_epoch}" + "\n"
    out_put(perf, file_name)


@torch.no_grad()
def val(model, dataloader, use_gpu):
    model.eval()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pred_label = []
    true_label = []
    
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    for ii, (data, mask, target) in enumerate(dataloader):
        data, mask = data.to(device), mask.to(device)
        target = target.to(device)

        # # with torch.no_grad():
        # ccl_reps = model.gen_f_reps(data, mask)
        # output = model.predictor(ccl_reps)
        output = model(data, mask)

        loss = criterion(output, target)
        loss_meter.add(loss.item())

        _, pred = output.data.topk(1, dim=1)
        pred = pred.t().squeeze()

        pred_label.append(pred.detach().cpu())
        true_label.append(target.detach().cpu())

    pred_label = torch.cat(pred_label, 0)
    true_label = torch.cat(true_label, 0)


    val_f1 = f1_score(true_label, pred_label, average='weighted', zero_division=0)
    val_accuracy = accuracy_score(true_label, pred_label)
    val_precision = precision_score(true_label, pred_label, average='weighted', zero_division=0)
    val_recall = recall_score(true_label, pred_label, average='weighted', zero_division=0)

    model.train()

    return loss_meter.value()[0], val_f1, val_accuracy, val_precision, val_recall

