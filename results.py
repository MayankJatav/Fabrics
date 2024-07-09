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
        results_folder = "results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        with open(f"{results_folder}/{filename}.json", "w") as file:
            file.write(json.dumps(data))
        print(f"Accuracy and Loss saved in {filename}.json successfully")