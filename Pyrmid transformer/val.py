import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import getpass

from models_mae import Mae
from tensorboardX import SummaryWriter
from dataset import dataload,testload
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
    def __init__(self,val_load,cfg):

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

    def train(self):

        state = load_checkpoint(os.path.join(self.log_path, "latest"))
        epoch = state["epoch"] + 1

        self.model.load_state_dict(state["model"])
        self.best_acc = state["best_acc"]

        self.model.eval()
        with torch.no_grad():

            for idx, (image, target) in enumerate(self.val_load):
                image = image.float()

                loss, pred, mask =self.model(image)

                image=image.permute(0,2,3,1).numpy()
                image=(image*255).astype(np.uint8)
                plt.imshow(image[0])  # 转换通道顺序

                plt.show()
                B,N,D=pred.shape
                pred=pred.reshape(B,8,8,4,4,3).permute(0,5,1,3,2,4)
                pred=pred.reshape(B,3,32,32)
                xx=(pred[0].permute(1,2,0).numpy()*255).astype(np.uint8)
                plt.imshow(xx)  # 转换通道顺序

                plt.show()
cfg=CFG
val_dataload=testload(64)
x=trainer(val_dataload,cfg)
x.train()