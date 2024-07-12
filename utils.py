import json

classes = ['Acetate', 'Acrylic', 'Angora', 'Cashmere', 'Chenille cotton', 'Cotton', 'Dralon', 'Elastane', 'Erion', 'Fleece', 'LI', 'Leather', 'Linen', 'Lut', 'Lycra', 'Metallic fibers', 'Micropoly', 'Modal', 'Naylon', 'Nylon', 'PES', 'Polyamide', 'Polyester', 'RAY', 'Rayon', 'Repreve', 'SP', 'Saduk Silk', 'Satin Silk', 'Silk', 'Spandex', 'Supplex Nylon', 'Synthetic', 'Viscose', 'Wool']

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