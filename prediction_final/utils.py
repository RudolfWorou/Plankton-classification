import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from prediction_final.datasets import *
import os
import time
import csv


##################################################################################
#Training

def train_model(model, optimizer, scheduler, loss, train_loader, eval_loader, name, mode, nb_epochs=20, logs="./logs", plotting=False, start=-1):
    """
    function to train a model
    """
    if mode == "train":
        logdir = generate_unique_logpath(logs, name)
    else: logdir = logs+"/"+name
    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    model_checkpoint = ModelCheckpoint(logdir + "/best_model.pt", model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loss_list = []
    eval_loss_list = []
    train_score_list = []
    eval_score_list = []
    times = []
    #set up variables and classes

    for e in range(start+1, start+nb_epochs+1):
        print(f"Epoch {e}")
        start_time = time.time()
        train_loss, train_score = _train_epoch(model, train_loader, device, loss, optimizer)
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        eval_loss, eval_score = _eval_epoch(model, eval_loader, device, loss)

        model_checkpoint.update(eval_loss, e)

        scheduler.step(eval_loss)

        #log the metrics, images...
        eval_loss_list.append(eval_loss)
        eval_score_list.append(eval_score)
        train_loss_list.append(train_loss)
        train_score_list.append(train_score)
        print('epoch: ', e, ' ---> train loss: ', train_loss, ' ---> eval loss: ', eval_loss, ' ---> train score: ', train_score, ' ---> test score: ', eval_score,' ---> computation time: ' ,epoch_time)

        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, e)
        tensorboard_writer.add_scalar('metrics/train_score',  train_score, e)
        tensorboard_writer.add_scalar('metrics/eval_loss', eval_loss, e)
        tensorboard_writer.add_scalar('metrics/eval_score',  eval_score, e)
        tensorboard_writer.add_scalar('metrics/epoch_time', epoch_time, e)
        tensorboard_writer.add_scalar('metrics/learning_rate', scheduler.optimizer.param_groups[0]['lr'], e)

def _train_epoch(model, train_loader, device, f_loss, optimizer):
    model.train()
    loss = 0.
    score = 0.
    n = 0

    for X, y in tqdm(train_loader):
#    for X, y in train_loader:

        #send data to gpu
        X, y = X.to(device), y.to(device)

        #reset gradient
        optimizer.zero_grad()

        #forward pass
        y_pred = model(X)
        loss_value = f_loss(y_pred, y)

        #adding up
        loss += loss_value.item()*len(y)

        #backward pass
        loss_value.backward()

        model.penalty().backward()

        #update params
        optimizer.step()

        #scoring
        y_pred = y_pred.argmax(dim=1)
        score += f1_score( y.cpu(), y_pred.cpu(), average="macro")*len(y)
        n += len(y)
    return round(loss/n,6), round(score/n,3)


def _eval_epoch(model, loader,device, f_loss ):
    #evaluation on the validation set
    model.eval()
    n=0
    loss = 0.
    score = 0.

    with torch.no_grad():
        for X, y in tqdm(loader):

            X, y = X.to(device), y.to(device)

            #predict and get loss
            y_pred = model(X)
            loss += f_loss(y_pred, y).item()*len(y)
            y_pred = y_pred.argmax(dim=1)

            score += f1_score(y.cpu(), y_pred.cpu(), average="macro")*len(y)
            n += len(y)

    return round(loss/n,6), round(score/n,3)

#######################################################
#Model saving

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            break
        i = i + 1
    os.mkdir(log_path)
    print("Logging to {}".format(log_path))
    return log_path

class ModelCheckpoint:
    """
    class ot save the model and epoch number
    """

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss, e):
        if (self.min_loss is None):
            res = {"model":self.model.state_dict(),
                    "epoch":e}
        elif (loss < self.min_loss):
            print("Saving a better model")
            res = torch.load(self.filepath)

            res['model'] = self.model.stat_dict()
            self.min_loss = loss
        else:
            res = torch.load(self.filepath)
        res['epoch'] = e
        torch.save(res, self.filepath)

#######################################################
#Resume training

def load_model(path, model):
    checkpoint = torch.load(path)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model'])

    return model, checkpoint['epoch']


######################################################
#Prediction

def  predict(path, model, test_path, final_size=64):
    model, _ = load_model(path, model)

    test_dataset = SRTestDataset(test_path, final_size=final_size)
    print(final_size)
    print(len(test_dataset))
    test_dataloader = torch.utils.data.dataloader.DataLoader(
                        test_dataset,
                        batch_size=64,
                        num_workers=4,
                        shuffle=False)

    test_predictions = {"imgname": [], "label":[]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval()
        model.to(device)
        for image, names in tqdm(test_dataloader):
            image = image.to(device)
            predictions = model(image).argmax(dim=1)
            for i in range(len(names)):
                test_predictions['label'].append(predictions[i].item())
                test_predictions["imgname"].append(names[i])

    #saving results
    with open("predictions_" + path.split("/")[-1].split(".")[-2] +".csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(test_predictions.keys())
        writer.writerows(zip(*test_predictions.values()))
