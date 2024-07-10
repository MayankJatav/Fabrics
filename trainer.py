import torch
import torch.optim as optim
from tqdm.auto import tqdm
from results import Results as results
from utils import Utilities as utils
import os

class Trainer:

    def __init__(self, model, loss_fn, optimizer, epochs, train_loader, val_loader=None, device=None, log_results_file=None, save_model_file=None):
        self.loss_fn = loss_fn
        self.optimizer: optim.Optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.log_results_file = log_results_file
        self.save_model_file = save_model_file
        if device is None:
                self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        print(f"Using {self.device} device")
        self.model = model.to(self.device)
        if self.save_model_file:
            torch.save(self.model, self.save_model_file + f"-epoch-{0}")

    def train_one_epoch(self):
        print("Training")
        train_running_loss = 0.0
        train_running_correct = 0
        self.model.train()
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            train_running_loss += loss.item()
            train_running_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            loss.backward()
            self.optimizer.step()
        epoch_loss = train_running_loss / len(self.train_loader)
        epoch_acc = 100 * (train_running_correct / len(self.train_loader.dataset))
        return epoch_loss, epoch_acc

    def validate_one_epoch(self):
        print("Validation")
        self.model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_running_loss += loss.item()
                val_running_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
        epoch_loss = val_running_loss / len(self.val_loader)
        epoch_acc = 100 * (val_running_correct / len(self.val_loader.dataset))
        return epoch_loss, epoch_acc

    def train(self):
        for epoch in range(self.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc = self.train_one_epoch()
            val_epoch_loss, val_epoch_acc = self.validate_one_epoch()
            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.val_loss.append(val_epoch_loss)
            self.val_acc.append(val_epoch_acc)
            if self.log_results_file:
                results.save_acc_loss_in_file(self.log_results_file, self.train_acc, self.train_loss, self.val_acc, self.val_loss)
            if self.save_model_file:
                os.rename(self.save_model_file + f"-epoch-{epoch}", self.save_model_file + f"-epoch-{epoch+1}")
                torch.save(self.model, self.save_model_file + f"-epoch-{epoch+1}")
            print(f"Training loss: {train_epoch_loss:.3f}, Training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {val_epoch_loss:.3f}, Validation acc: {val_epoch_acc:.3f}")
            print()
        print("Training Complete")
        return self.train_acc, self.train_loss, self.val_acc, self.val_loss
