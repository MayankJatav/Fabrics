import json

classes = ['Cotton', 'Polyester', 'Wool']

class Utilities:

    def class_to_index(className):
        return classes.index(className)

    def index_to_class(index):
        return self.classes[index]

    def get_classes():
        return classes

    def get_config():
        with open('config.json') as file:
            config = json.load(file)
            return config
    
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False