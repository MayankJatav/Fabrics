import torch
from transformers import MobileViTImageProcessor, MobileViTForImageClassification, MobileViTFeatureExtractor

class MobileVIT(torch.nn.Module):
    def __init__(self, device=None):
        super(MobileVIT, self).__init__()
        self.device = device
        # self.feature_extractor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        # self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        self.feature_extractor = MobileViTFeatureExtractor.from_pretrained("./cache/Tokenizer/MobileViTModel")
        self.model = MobileViTForImageClassification.from_pretrained("./cache/model/MobileViTModel")
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(640, 3),
            torch.nn.Softmax()
        )

    def forward(self, input):
        inputs = self.feature_extractor(images=input, return_tensors="pt")
        # inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.logits