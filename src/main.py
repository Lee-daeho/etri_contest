import train
from train import train
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Per-Session experiment')
    parser.add_argument('--epoch', '-e', default=100, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=128, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=1e-4, help='Learn rate in training', type=float)
    parser.add_argument('--weight_decay', '-decay', default=1e-4, help='Weight Decay in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    parser.add_argument('--modal', '-m', default='text', help='Type of data to train', type=str)
    parser.add_argument('--l_type', default='valence', help='Valence or arousal', type=str)
    parser.add_argument('--pretrain', default='True', help='Use pretrained CNN', type=str)

    args = parser.parse_args()

    use_gpu = True if args.gpu == 'True' else False
    pretrain = True if args.pretrain == 'True' else False

    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')

    if not os.path.exists(f'./results/{args.modal}/'):
        os.mkdir(f'./results/{args.modal}/')

    # 5 Fold validation (k: fold)
    for k in range(1, 6):
        train(modal=args.modal, epochs=args.epoch, lr=args.learn_rate, decay=args.weight_decay,
              use_gpu=use_gpu,
              file_name=f'./results/{args.modal}/{args.modal}_{args.l_type}_k{k}/{args.modal}_{args.label}_k{k}',
              batch_size=args.batch_size, k=k, l_type=args.l_type, pretrain=pretrain)


if __name__ == '__main__':
    main()
