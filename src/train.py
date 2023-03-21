import torch
import os

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

from src.dataset import KEMDyDataset

from torchnet import meter


def train(modal, k, l_type, epochs, lr, decay, batch_size, file_name, use_gpu=False, pretrain=True):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    directory = file_name.split('/')[-2]
    if not os.path.exists(f'./results/{modal}/' + directory):
        os.mkdir(f'./results/{modal}/' + directory)

    train_data = KEMDyDataset(modal=modal, k=k, kind='train', l_type=l_type)
    val_data = KEMDyDataset(modal=modal, k=k, kind='val', l_type=l_type)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model Init
    # model = Example().to(device)

    # Optimization Settings
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    # meters
    # loss_meter = meter.AverageValueMeter()

    # Training
    # for epoch in range(epochs)

    # @torch.no_grad()
    # def val(model, dataloader, use_gpu, modal)
