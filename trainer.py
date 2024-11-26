import torch
import torch.optim as optim
from tqdm.auto import tqdm
from results import Results as results
from utils import Utilities as utils
import os
from sklearn.metrics import precision_score, recall_score

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
        self.train_precision = []
        self.train_recall = []
        self.val_precision = []
        self.val_recall = []
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
        all_labels = []
        all_preds = []
        self.model.train()
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).logits
            loss = self.loss_fn(outputs, labels)
            train_running_loss += loss.item()
            train_running_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            all_labels.extend(labels.argmax(1).cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            loss.backward()
            self.optimizer.step()
        epoch_loss = train_running_loss / len(self.train_loader)
        epoch_acc = 100 * (train_running_correct / len(self.train_loader.dataset))
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall

    def validate_one_epoch(self):
        print("Validation")
        self.model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).logits
                loss = self.loss_fn(outputs, labels)
                val_running_loss += loss.item()
                val_running_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                all_labels.extend(labels.argmax(1).cpu().numpy())
                all_preds.extend(outputs.argmax(1).cpu().numpy())
        epoch_loss = val_running_loss / len(self.val_loader)
        epoch_acc = 100 * (val_running_correct / len(self.val_loader.dataset))
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall

    def train(self):
        for epoch in range(self.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc, train_epoch_precision, train_epoch_recall = self.train_one_epoch()
            val_epoch_loss, val_epoch_acc, val_epoch_precision, val_epoch_recall = self.validate_one_epoch()
            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.train_precision.append(train_epoch_precision)
            self.train_recall.append(train_epoch_recall)
            self.val_loss.append(val_epoch_loss)
            self.val_acc.append(val_epoch_acc)
            self.val_precision.append(val_epoch_precision)
            self.val_recall.append(val_epoch_recall)
            if self.log_results_file:
                results.save_acc_loss_precision_recall_in_file(self.log_results_file, self.train_acc, self.train_loss, self.train_precision, self.train_recall, self.val_acc, self.val_loss, self.val_precision, self.val_recall)
            if self.save_model_file:
                os.rename(self.save_model_file + f"-epoch-{epoch}", self.save_model_file + f"-epoch-{epoch+1}")
                torch.save(self.model, self.save_model_file + f"-epoch-{epoch+1}")
            print(f"Training loss: {train_epoch_loss:.3f}, Training acc: {train_epoch_acc:.3f}, Precision: {train_epoch_precision:.3f}, Recall: {train_epoch_recall:.3f}")
            print(f"Validation loss: {val_epoch_loss:.3f}, Validation acc: {val_epoch_acc:.3f}, Precision: {val_epoch_precision:.3f}, Recall: {val_epoch_recall:.3f}")
            print()
        print("Training Complete")
        return self.train_acc, self.train_loss, self.val_acc, self.val_loss
