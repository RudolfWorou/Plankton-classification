import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import argparse

from prediction_final.utils import *
from prediction_final.models import *
from prediction_final.datasets import *

def main(args):
    #define parameters for the training
    nb_class = 86
    resize= args.size
    padding = False
    batch_size = 32

    #create model here
    model = MyResNet152(images_size=resize, freeze=args.freeze, l2=0)

    if args.mode!="test":
        train_dataset = AugmentedDataset(args.path, with_test=True, final_size=resize, minimum=2000, maximum=1000000)
        eval_dataset = SREvalDataset( *train_dataset.get_test_set(), final_size=resize)
        len_train = len(train_dataset)
        len_eval = len(eval_dataset)
        print(f'Train data set contains {len_train} image(s)')
        print(f'Eval data set contains {len_eval} image(s)')
        train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, weights=False)
        eval_dataloader = get_dataloader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, weights=False)
        #define datasets and dataloaders

        if args.mode == "train":
            epochs_done = -1
        elif args.mode == "resume":
            model, epochs_done =  load_model(args.model_path, model)
            print("Successfully loaded model")
        #load model if resuming the training

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=args.l2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
        #define loss, optimizer and scheduler

        train_model(model, optimizer, scheduler, loss_function, train_dataloader, eval_dataloader, args.model, args.mode, nb_epochs=args.epochs, logs=args.logdir, start=epochs_done)

    elif args.mode == "test":
        predict(args.model_path, model, args.path, final_size=resize)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
'mode',
choices= ["train", "resume", "test"],
help='Mode of the script: train, resume, test',
)
parser.add_argument(
'model_path',
help='Path to model',
nargs='?'
)
parser.add_argument(
'path',
help='Path to training/test data',
)

parser.add_argument(
'--epochs',
type=int,
default=50,
help='Number of epochs',
)
parser.add_argument(
'--ratio',
type=float,
default=0.2,
help='Ratio of the evaluation set',
)
parser.add_argument(
'--logdir',
type=str,
default="./logs",
help='The directory in which to store the logs'
)
parser.add_argument(
'--model',
type=str,
default="MyModel",
help='Name of the model'
)
parser.add_argument(
'--l2',
type=float,
default=0.,
help="Value for L2 regularization"
)
parser.add_argument(
'--lr',
type=float,
default=1e-3,
help="Value for the learning rate"
)
parser.add_argument(
'--size',
type=int,
default=224,
help="Image size for model"
)
parser.add_argument(
'--freeze',
type=bool,
default=False,
help="Freezing the weights of the pretrained layers"
)

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    main(args)
