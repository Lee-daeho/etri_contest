from src import train
from src.train import train
import argparse
import os
import wandb



def main():
    parser = argparse.ArgumentParser(description='Per-Session experiment')
    parser.add_argument('--epoch', '-e', default=40, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=16, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=1e-5, help='Learn rate in training', type=float)
    parser.add_argument('--weight_decay', '-decay', default=0.9, help='Weight Decay in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    parser.add_argument('--modal', '-m', default='text', help='Type of data to train', type=str)
    parser.add_argument('--l_type', default='emotion', help='Emotion or valence or arousal', type=str)

    args = parser.parse_args() 

    use_gpu = True if args.gpu == 'True' else False



    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')

    if not os.path.exists(f'./results/{args.modal}/'):
        os.mkdir(f'./results/{args.modal}/')

    # 5 Fold validation (k: fold)
    for k in range(1, 6):

        # wandb.init(
        #     project='etri_contest',
        #     entity='jspirit01',
        #     config = {
        #         'learning_rate': args.learn_rate,
        #         'epochs': args.epoch,
        #         'batch_size': args.batch_size,
        #         'modal': 'wav'
        #     },
        #     name= f'wav2vec_fold{k}'
        # )

        train(modal=args.modal, epochs=args.epoch, lr=args.learn_rate, decay=args.weight_decay,
              use_gpu=use_gpu,
              file_name=f'./results/{args.modal}/{args.modal}_{args.l_type}_k{k}/{args.modal}_{args.l_type}_k{k}',
              batch_size=args.batch_size, k=k, l_type=args.l_type)

        # wandb.finish()

if __name__ == '__main__':
    main()
