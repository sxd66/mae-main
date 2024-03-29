import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import getpass

from models_mae import Mae
from tensorboardX import SummaryWriter
from dataset import dataload
from utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
class CFG:
    lr=0.008
    experiment_name="mae"

class trainer():
    def __init__(self,train_load,val_load,cfg):
        self.train_load=train_load
        self.val_load=val_load
        self.model=Mae()
        experiment_name=cfg.experiment_name
        self.optimizer = self.init_optimizer(cfg.lr,0.9)
        self.best_acc = -1

        self.loss=nn.CrossEntropyLoss()
        self.log_path = os.path.join('../log', experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))


        self.maxepoch=200
        self.check_fre=5


    def init_optimizer(self,lr,mount):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=mount,
       )
        return optimizer
    def log(self,  epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()

    def train(self,resume=False):
        epoch=1

        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1

            self.model.load_state_dict(state["model"])
            self.best_acc = state["best_acc"]

        while epoch<self.maxepoch :
            self.train_epoch(epoch)
            epoch+=1
    def train_epoch(self,epoch):
        #lr=adjust_learning_rate(epoch)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),


        }
        num_iter = len(self.train_load)
        pbar = tqdm(range(num_iter))
        self.model.train()
        adjust_learning_rate(epoch,0.8,self.optimizer)
        for index,data in enumerate(self.train_load):
            msg=self.train_iter(data,epoch,train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()
        if(epoch%self.check_fre==0):
            losses=validate(self.val_load,self.model)
            #if( losses>self.best_acc ):
            #    self.best_acc=losses

        log_dict = OrderedDict(
            {

                "train_loss": train_meters["losses"].avg,

            }
        )
        self.log(epoch, log_dict)

        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        save_checkpoint(state, os.path.join(self.log_path, "latest"))

    def train_iter(self,data,epoch,train_meters):

        train_start_time = time.time()
        image,target=data
        train_meters["data_time"].update(time.time() - train_start_time)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image=image.float().to(device)
        target = target.to(device)

        self.model.to(device)


        self.optimizer.zero_grad()
        lost, pred, mask=self.model(image)



        lost.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)

        train_meters["losses"].update(lost.cpu().detach().numpy().mean(), batch_size)


        learning_rate = self.optimizer.param_groups[0]['lr']
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Lr:{}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            learning_rate
        )
        return msg


cfg=CFG
train_dataload,val_dataload=dataload(64,64)
x=trainer(train_dataload,val_dataload,cfg)
x.train(resume=True)






















