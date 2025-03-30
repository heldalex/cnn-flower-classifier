import os
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from src.cnn import *
from src.eval.evaluate import eval_fn, accuracy
from src.training import train_fn
from src.data_augmentations import *


def main(data_dir,
         torch_model,
         train_augmentation=None,
         val_augmentation=None,
         num_epochs=50,
         batch_size=32,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
         model_optimizer=torch.optim.AdamW,
         save_model_str=None,
         use_all_data_to_train=False,
         exp_name=''):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param train_augmentation:
    :param val_augmentation:
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param save_model_str: path of saved models (str)
    :param use_all_data_to_train: indicator whether we use all the data for training (bool)
    :param exp_name: experiment name (str)
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Process augmentation args
    train_transform = train_augmentation if train_augmentation else transforms.ToTensor()
    val_transform = val_augmentation if val_augmentation else transforms.ToTensor()

    # Load datasets with separate transforms
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)  # Use val augmentations for test

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []

    if use_all_data_to_train:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                  batch_size=batch_size,
                                  shuffle=True)
        logging.warning('Training with all the data (train, val and test).')
    else:
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)

    model = torch_model(input_shape=input_shape,
                        num_classes=len(train_data.classes)).to(device)

    optimizer = model_optimizer(model.parameters(),
                                lr=learning_rate,
                                weight_decay=0.0001,
                                eps=1e-8)

    warmup_epochs = 5  # First 5 epochs for linear warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    # Info about the model being trained
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # More info
    logging.info(f"Batch: {batch_size}, LR: {learning_rate}, Optimizer: {model_optimizer.__name__}")

    # Lists to store metrics
    train_scores = []
    val_scores = []

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        if epoch < warmup_epochs:
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_score, train_loss = train_fn(model,
                                           optimizer,
                                           train_criterion,
                                           train_loader,
                                           device)
        #train_scores.append(train_score)  # Store training accuracy
        logging.info('Train accuracy: %f', train_score)

        if epoch >= warmup_epochs:
            scheduler.step()

        logging.info(f'Epoch {epoch + 1}: LR = {optimizer.param_groups[0]["lr"]:.2e}')

        if not use_all_data_to_train:
            test_score = eval_fn(model, val_loader, device)
            #val_scores.append(test_score)  # Store validation accuracy
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)

        '''
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, num_epochs + 1)
        plt.plot(epochs_range, train_scores, label='Training Accuracy', marker='o')
        if not use_all_data_to_train:
            plt.plot(epochs_range, val_scores, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy Over Epochs ({exp_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_model_str or 'models', f'{exp_name}_accuracy_plot.png'))  # Save plot
        plt.show()  # Display plot (optional, remove if running non-interactively)
        '''

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + str(int(time.time())))
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100*np.mean(score))+'%')
        logging.info('Accuracy of model at final epoch: ' + str(100*score[-1])+'%')


if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {
        'cross_entropy': lambda: torch.nn.CrossEntropyLoss,  # Base option
        'cross_entropy_ls': lambda: torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Matches main
    }
    opti_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }

    cmdline_parser = argparse.ArgumentParser('DL WS24/25 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='CnnModel',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_and_colour_jitter',
                                help='Training augmentation (from data_augmentations.py)')
    cmdline_parser.add_argument('--val-augmentation',
                                default='resize_to_64x64',
                                help='Validation augmentation (from data_augmentations.py)')
    cmdline_parser.add_argument('-e', '--epochs',
                                default=100,  # number of epochs
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=32,  # batch size
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=0.003,  # learning rate
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy_ls',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adamw',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-a', '--use-all-data-to-train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        train_augmentation=eval(args.data_augmentation),
        val_augmentation=eval(args.val_augmentation),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train
    )
