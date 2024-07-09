import json
import dataloader
import model
import trainer
from results import Results as results
from utils import Utilities as utils
import torch
import torchvision
from torch.utils.data import random_split

if __name__ == '__main__':
    config = utils.get_config()
    dataset_path = config['dataset_path']
    epochs = config['epochs']
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    results_file = config['results_file']
    results_image = config['results_image']
    save_model_file = config['save_model_file']
    current_run_dir = results.create_new_result_dir()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224))
        ])

    dataset = dataloader.FabricDataset(dataset_path, transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    train_model = model.Model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=learning_rate)

    train_runner = trainer.Trainer(train_model,
                                    loss_fn, 
                                    optimizer, 
                                    epochs, 
                                    train_loader, 
                                    val_loader, 
                                    log_results_file=f"{current_run_dir}/{results_file}",
                                    save_model_file=f"{current_run_dir}/{save_model_file}"
                                    )
    result = train_runner.train()

    results.save_acc_loss_graph(f"{current_run_dir}/{results_image}", *result)