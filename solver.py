import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from tqdm import tqdm
from models.custom_model import CustomModel
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, valid_loader, test_loader, args):
        """Initialize configurations."""
        self.args = args
        self.model_name = '{}.pth'.format(self.args.model_name)
        self.device = self.args.device

        # Define the model

        self.model = CustomModel(self.args).to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader


        # load a pretrained model
        if self.args.resume_train or self.args.mode in ['test','evaluate']:
            self.load_model()
        
        if(self.args.mode == "train"):
            # Choose optimizer 
            if self.args.opt == "SGD":
                self.optimizer = optim.SGD(self.model.params, lr=self.args.lr)
            elif self.args.opt == "Adam":
                self.optimizer = optim.Adam(self.model.params, lr=self.args.lr)
            elif self.args.opt == "AdamW":
                self.optimizer = optim.AdamW(self.model.params, lr=self.args.lr)

            self.epochs = self.args.epochs
            self.writer = SummaryWriter(self.args.writer_path + '/' + self.args.model_name)

    def save_model(self, epoch):
        # if you want to save the model
        checkpoint_name = "epoch" + str(epoch) + "_" + self.model_name
        check_path = os.path.join(self.args.checkpoint_path, checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.model.load_state_dict(torch.load(check_path, map_location=torch.device(self.device)))
        print("Model loaded!", flush=True)
    
    def train(self):
        self.model.train()
        self.train_loss = []
        self.val_loss = []
        early_stopping = EarlyStopping(patience=2, verbose=True)
        for epoch in range(self.epochs):
            print(f"\nEPOCH {epoch+1} of {self.epochs}", flush=True)
            running_loss = 0.0
            # start timer and carry out training and validation
            start = time.time()
            print('Solver Training', flush=True)
            train_loss_list = []
            
            # initialize tqdm progress bar
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))

            for i, data in enumerate(prog_bar):
                self.optimizer.zero_grad()

                # compute loss
                loss = self.model(...)
                train_loss_list.append[loss]
                self.optimizer.step()
                
                prog_bar.set_description(desc=f"Loss: {loss:.4f}")

               
                if i % self.args.print_every == self.args.print_every - 1:  
                    print("Loss:", loss, flush=True)
                running_loss = 0.0

            val_loss_list = self.validate()
            
            print(f"Epoch #{epoch+1} train loss: {sum(train_loss_list)/len(self.train_loader):.3f}", flush=True)   
            print(f"Epoch #{epoch+1} validation loss: {sum(val_loss_list)/len(self.valid_loader):.3f}", flush=True)  

            self.train_loss.append(sum(train_loss_list)/len(self.train_loader));
            self.val_loss.append(sum(val_loss_list)/len(self.valid_loader));

            self.writer.add_scalar('validation loss',
                        sum(val_loss_list)/len(self.valid_loader),epoch)
            self.writer.add_scalar('train loss',
                        sum(train_loss_list)/len(self.train_loader),epoch)
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}", flush=True)
            self.save_model(epoch)

            # perform early stopping here
            if self.args.early_stopping:
                print("Early stopping", flush=True)
                break
        
        self.evaluate(epoch)   
        self.writer.flush()
        self.writer.close()
        print('Finished Training', flush=True)  

    def validate(self):
        print('Validating')
        val_itr = 0
        val_loss_list = []
        # initialize tqdm progress bar
        prog_bar = tqdm(self.valid_loader, total=len(self.valid_loader))
        loss_value = 0
        self.model.eval()
        for i, data in enumerate(prog_bar):
            with torch.no_grad():
                # compute loss
                loss_value = ...
                val_loss_list.append(loss_value)
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}\n\n")
        self.model.train()
        return val_loss_list
    
    def test(self, img_count=5):
        print("Testing", flush=True)
        i = 0
        for data in self.test_loader:
            if(i==img_count):
                break
            images, targets = data
            self.model.eval()
            # perform test here

            i+=1
            
    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            # Eval model here
            print("Eval metrics: ", flush=True)
        self.model.train()
    

    def debug(self):
        print("Debug")