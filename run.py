import json
import dataloader
from models.InceptionModel import InceptionModel
from models.MobileNetModel import MobileNetModel
import trainer
from results import Results as results
from utils import Utilities as utils
import torch
import torchvision
from torchsummary import summary

if __name__ == '__main__':
    config = utils.get_config()
    dataset_path = config['dataset_path']
    dataset_name = config['dataset_name']
    epochs = config['epochs']
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    results_file = config['results_file']
    results_image = config['results_image']
    save_model_file = config['save_model_file']
    saved_weights = config['saved_weights']
    model_input_shape = tuple(config['model_input_shape'])
    model_name = config['model_name']
    current_run_dir = results.create_new_result_dir()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(model_input_shape)
        ])

    if dataset_name.lower() == "FabricsDataset".lower():
        train_loader, val_loader = dataloader.get_fabrics_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size)
    elif dataset_name.lower() == "FabricsOCTDataset".lower():
        train_loader, val_loader = dataloader.get_fabrics_oct_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size)
    elif dataset_name.lower() == "TextileNetDataset".lower():
        train_loader, val_loader = dataloader.get_textilenet_dataset_dataloder(dataset_path, transform, train_size, val_size, batch_size)
    else:
        assert False, "Dataset name should be either FabricsDataset or FabricsOCTDataset"
    
    if model_name == "mobilenet":
        train_model = MobileNetModel(pretrained=True)
    if model_name == "inceptionv3":
        train_model = InceptionModel(pretrained=True)
    if saved_weights:
        print("Loading Saved Weights:", saved_weights)
        train_model.load_state_dict(torch.load(saved_weights).state_dict())

    loss_fn = torch.nn.L1Loss()
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