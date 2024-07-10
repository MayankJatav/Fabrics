from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
from utils import Utilities as utils

class Results:
    
    def create_new_result_dir():
        config = utils.get_config()
        results_folder = config['results_dir']
        now = datetime.now()
        folder_name = now.strftime("%d-%m-%Y_%H-%M-%S")
        if not os.path.exists(f"{results_folder}/{folder_name}"):
            os.makedirs(f"{results_folder}/{folder_name}")
        return f"{results_folder}/{folder_name}"

    def save_acc_loss(acc, loss):
        config = utils.get_config()
        results_file = config['results_file']
        folder_name = Results.create_new_result_dir()
        data = {}
        data['acc'] = acc
        data['loss'] = loss
        with open(f"{folder_name}/{results_file}", "w") as file:
            file.write(json.dumps(data))
        print(f"Accuracy and Loss saved in {folder_name}/{results_file} successfully")

    def save_acc_loss_in_file(filename, train_acc, train_loss, val_acc, val_loss):
        data = {}
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss
        with open(f"{filename}", "w") as file:
            file.write(json.dumps(data))

    def save_acc_loss_graph(imagePath, train_acc, train_loss, val_acc, val_loss):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(train_acc, label='Train Accuracy')
        axes[0].plot(val_acc, label='Validation Accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[1].plot(train_loss, label='Train Loss')
        axes[1].plot(val_loss, label='Validation Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        plt.savefig(imagePath)

    def save_acc_loss_graph_from_file(resultsFile, imagePath):
        with open(f"{resultsFile}", "r") as file:
            data = json.load(file)
            train_acc = data['train_acc']
            train_loss = data['train_loss']
            val_acc = data['val_acc']
            val_loss = data['val_loss']
            Results.save_acc_loss_graph(imagePath, train_acc, train_loss, val_acc, val_loss)

    def combine_results(file1_path, file2_path):
        with open(file1_path, 'r') as file1:
            data1 = json.load(file1)
        with open(file2_path, 'r') as file2:
            data2 = json.load(file2)
        combined_data = {}
        for key in data1.keys():
            val1 = data1[key]
            val2 = data2[key]
            val = val1 + val2
            combined_data[key] = val
        return combined_data

    def combine_and_save_results(result1_path, result2_path, save_path):
        with open(save_path, 'w') as output_file:
            combined_data = Results.combine_results(result1_path, result2_path)
            js = json.dump(combined_data, output_file, indent=4)