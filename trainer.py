import torch.optim as optim
from tqdm.auto import tqdm

class Trainer:

    def __init__(self, model, loss_fn, optimizer, epochs, train_loader, val_loader=None):
        self.loss_fn = loss_fn
        self.optimizer: optim.Optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = 0.0
        self.correct = 0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def train_one_epoch():
        self.loss = 0.0
        self.correct = 0
        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = model(inputs)
            self.loss = self.loss_fn(outputs, labels)
            self.loss.backward()
            self.optimizer.step()
            self.loss += self.loss.item()
            self.correct = (outputs.argmax(1) == labels).sum().item()
        self.loss /= len(train_loader)
        acc = 100 * (self.correct / len(train_loader))
        return self.loss, acc

    def train():
        for epoch in range(self.epochs):
            model.train()
            train_epoch_loss, train_epoch_acc = train_one_epoch()
            model.eval()
            val_epoch_loss, val_epoch_acc = validate_one_epoch()
            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.val_loss.append(val_epoch_loss)
            self.val_acc.append(val_epoch_acc)
            print(f"Training loss: {train_epoch_loss:.3f}, Training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {val_epoch_loss:.3f}, Validation acc: {val_epoch_acc:.3f}")
