import json
import dataloader
from models.DeitModel import DeiTModel
from models.InceptionModel import InceptionModel
from models.MobileNetModel import MobileNetModel
from models.VGG16 import VGG16
import trainer
from models.MobileVIT import MobileVIT
from models.TextileNetPaper import TextileNetModel
from results import Results as results
from utils import Utilities as utils
import torch
import torchvision
from torch.utils.data import random_split
from torchsummary import summary

if __name__ == '__main__':
    config = utils.get_config()
    dataset_path = config['dataset_path']
    epochs = config['epochs']
    train_size = config['train_size']
    val_size = config['val_size']
    # test_size = config['test_size']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    results_file = config['results_file']
    results_image = config['results_image']
    save_model_file = config['save_model_file']
    saved_weights = config['saved_weights']
    model_input_shape = tuple(config['model_input_shape'])
    model_name = config['model_name']
    device = config['device']
    current_run_dir = model_name+"_"+results.create_new_result_dir()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(model_input_shape)
        ])

    dataset = dataloader.FabricDataset(dataset_path, transform)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    
    # if model_name == "mobilenet":
    #     train_model = MobileNetModel(pretrained=True)
    # if model_name == "inceptionv3":
    #     train_model = InceptionModel(pretrained=True)
    # if model_name == "deit":
    #     train_model = DeiTModel(pretrained=True)
    # if model_name == "mobilevit":
    #     train_model = MobileVIT(device=device)
    # if model_name == "vgg16":
    #     train_model = VGG16(pretrained=True)
    # if saved_weights:
    #     print("Loading Saved Weights:", saved_weights)
    #     train_model.load_state_dict(torch.load(saved_weights).state_dict())
    if model_name == "textilenet":
        pt_model = "vgg16"
        train_model = TextileNetModel(pt_model)
        train_runner = trainer.Trainer(train_model, loss_fn, optimizer, epochs, train_loader, val_loader, 
                                        log_results_file=f"{current_run_dir}/{pt_model}/{results_file}",
                                        save_model_file=f"{current_run_dir}/{pt_model}/{save_model_file}"
                                        )
        result = train_runner.train()
        results.save_acc_loss_graph(f"{current_run_dir}/{pt_model}/{results_image}", *result)
        
        pt_model = "resnet101"
        train_model = TextileNetModel(pt_model)
        train_runner = trainer.Trainer(train_model, loss_fn, optimizer, epochs, train_loader, val_loader, 
                                        log_results_file=f"{current_run_dir}/{pt_model}/{results_file}",
                                        save_model_file=f"{current_run_dir}/{pt_model}/{save_model_file}"
                                        )
        result = train_runner.train()
        results.save_acc_loss_graph(f"{current_run_dir}/{pt_model}/{results_image}", *result)

        pt_model = "inceptionv3"
        train_model = TextileNetModel(pt_model)
        train_runner = trainer.Trainer(train_model, loss_fn, optimizer, epochs, train_loader, val_loader, 
                                        log_results_file=f"{current_run_dir}/{pt_model}/{results_file}",
                                        save_model_file=f"{current_run_dir}/{pt_model}/{save_model_file}"
                                        )
        result = train_runner.train()
        results.save_acc_loss_graph(f"{current_run_dir}/{pt_model}/{results_image}", *result)

        pt_model = "mobilenetv2"
        train_model = TextileNetModel(pt_model)
        train_runner = trainer.Trainer(train_model, loss_fn, optimizer, epochs, train_loader, val_loader, 
                                        log_results_file=f"{current_run_dir}/{pt_model}/{results_file}",
                                        save_model_file=f"{current_run_dir}/{pt_model}/{save_model_file}"
                                        )
        result = train_runner.train()
        results.save_acc_loss_graph(f"{current_run_dir}/{pt_model}/{results_image}", *result)

        pt_model = "convnextbase"
        train_model = TextileNetModel(pt_model)
        train_runner = trainer.Trainer(train_model, loss_fn, optimizer, epochs, train_loader, val_loader, 
                                        log_results_file=f"{current_run_dir}/{pt_model}/{results_file}",
                                        save_model_file=f"{current_run_dir}/{pt_model}/{save_model_file}"
                                        )
        result = train_runner.train()
        results.save_acc_loss_graph(f"{current_run_dir}/{pt_model}/{results_image}", *result)

    # if model_name == "textilenet_resnet101":
    # if model_name == "textilenet_inceptionv3":
    # if model_name == "textilenet_mobilenetv2":
    # if model_name == "textilenet_convnextbase":
    # print(train_model)
    # summary(train_model, (3, model_input_shape[0], model_input_shape[1]))
