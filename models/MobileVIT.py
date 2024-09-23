import torch
from transformers import MobileViTImageProcessor, MobileViTForImageClassification

class MobileVIT(torch.nn.Module):
    def __init__(self):
        super(MobileVIT, self).__init__()
        self.feature_extractor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(640, 21),
            torch.nn.Softmax(),
        )

    def forward(self, input):
        inputs = self.feature_extractor(images=input, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits