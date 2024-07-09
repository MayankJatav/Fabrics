from datetime import datetime
import json
import os

class Results:
    
    def save_acc_loss(acc, loss):
        now = datetime.now()
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")
        data = {}
        data['acc'] = acc
        data['loss'] = loss
        with open('config.json') as file:
            config = json.load(file)
        results_folder = config['results_dir']
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        with open(f"{results_folder}/{filename}.json", "w") as file:
            file.write(json.dumps(data))
        print(f"Accuracy and Loss saved in {filename}.json successfully")

    def save_acc_loss_in_file(filename, train_acc, train_loss, val_acc, val_loss):
        data = {}
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss
        with open(f"{filename}", "w") as file:
            file.write(json.dumps(data))