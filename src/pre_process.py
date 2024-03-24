import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from pre_trained_models import get_pre_trained_image_model, get_pre_trained_text_model


# import matplotlib.pyplot as plt


class ProcessDataset(Dataset):
    def __init__(self, train_json_path, train_images_folder_path, device="cpu", mode="train"):
        self.mode = mode
        self.train_ignore = [163, 1687, 1769, 3683]

        self.train_json_data = self.read_json_data(train_json_path)
        self.pretrained_image_model = get_pre_trained_image_model(device)
        self.pretrained_text_model, self.tokenizer = get_pre_trained_text_model()

        self.train_images_folder_path = train_images_folder_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :])  # Keep only the first 3 channels (R, G, B)
        ])
        self.max_length = 512
        self.device = device
        self.all_targets = [
            "Logos",
            "Repetition",
            "Obfuscation, Intentional vagueness, Confusion",
            "Reasoning",
            "Justification",
            "Slogans",
            "Bandwagon",
            "Appeal to authority",
            "Flag-waving",
            "Appeal to fear/prejudice",
            "Simplification",
            "Causal Oversimplification",
            "Black-and-white Fallacy/Dictatorship",
            "Thought-terminating cliché",
            "Distraction",
            "Misrepresentation of Someone's Position (Straw Man)",
            "Presenting Irrelevant Data (Red Herring)",
            "Whataboutism",
            "Ethos",
            "Glittering generalities (Virtue)",
            "Ad Hominem",
            "Doubt",
            "Name calling/Labeling",
            "Smears",
            "Reductio ad hitlerum",
            "Pathos",
            "Exaggeration/Minimisation",
            "Loaded Language",
            "Transfer",
            "Appeal to (Strong) Emotions"
        ]

    def __len__(self):
        return len(self.train_json_data)

    def __getitem__(self, idx):

        if (self.mode == "train" and idx in [163, 1687, 1769, 3683]):
            idx = 1000

        id = self.train_json_data[idx]['id']
        image_name = self.train_json_data[idx]['image']
        text_content = self.train_json_data[idx]['text']
        labels = self.train_json_data[idx]['labels']
        raw_gt = self.get_ground_truth(labels)
        gt_tensor = self.convert_gt_to_tensor(raw_gt)
        image_tensor = self.read_image(image_name)
        image_features = self.pretrained_image_model(image_tensor.to(self.device)).squeeze()
        text_features = self.get_text_tensor(text_content)

        if self.mode == "train":
            return image_features, text_features, gt_tensor
        else:
            return image_features, text_features, gt_tensor, id

    def read_image(self, image_name):
        image = Image.open(f"{self.train_images_folder_path}/{image_name}")
        image_tensor = self.transform(image)
        image_tensor = image_tensor.reshape(
            (1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))
        return image_tensor

    def get_text_tensor(self, text):
        tokens = self.tokenizer(text, truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')
        tokens.to(self.device)
        with torch.no_grad():
            outputs = self.pretrained_text_model(**tokens)

        features = outputs.last_hidden_state.mean(dim=1)
        return features.squeeze()

    def convert_gt_to_tensor(self, gt):
        categories = self.all_targets
        num_categories = len(categories)
        multi_label_ground_truth = [gt]

        # Create a tensor for multi-label classification
        tensor = np.zeros((len(multi_label_ground_truth), num_categories))
        for i, labels in enumerate(multi_label_ground_truth):
            indices = [categories.index(label) for label in labels]
            tensor[i, indices] = 1

        tensor = torch.tensor(tensor)
        return tensor.squeeze()
        # return tensor

    def get_ground_truth(self, labels):
        ground_truth = {}
        for label in labels:
            if (label in ["Name calling/Labeling", "Doubt", "Smears", "Reductio ad hitlerum"]):
                ground_truth[label] = 1
                ground_truth["Ethos"] = 1
                ground_truth["Ad Hominem"] = 1

            if (label in ["Bandwagon", "Appeal to authority"]):
                ground_truth[label] = 1
                ground_truth["Ethos"] = 1
                ground_truth["Logos"] = 1
                ground_truth["Justification"] = 1

            if (label in ["Glittering generalities (Virtue)"]):
                ground_truth[label] = 1
                ground_truth["Ethos"] = 1

            if (label in ["Transfer"]):
                ground_truth[label] = 1
                ground_truth["Ethos"] = 1
                ground_truth["Pathos"] = 1

            if (label in ["Appeal to (Strong) Emotions", "Exaggeration/Minimisation",
                          "Loaded Language"]):
                ground_truth[label] = 1
                ground_truth["Pathos"] = 1

            if (label in ["Flag-waving", "Appeal to fear/prejudice"]):
                ground_truth[label] = 1
                ground_truth["Pathos"] = 1
                ground_truth["Logos"] = 1
                ground_truth["Justification"] = 1

            if (label in ["Slogans"]):
                ground_truth[label] = 1
                ground_truth["Justification"] = 1
                ground_truth["Logos"] = 1

            if (label in ["Repetition", "Obfuscation, Intentional vagueness, Confusion"]):
                ground_truth[label] = 1
                ground_truth["Logos"] = 1

            if (label in ["Misrepresentation of Someone's Position (Straw Man)",
                          "Presenting Irrelevant Data (Red Herring)"]):
                ground_truth[label] = 1
                ground_truth["Logos"] = 1
                ground_truth["Distraction"] = 1
                ground_truth["Reasoning"] = 1

            if (label in ["Whataboutism"]):
                ground_truth[label] = 1
                ground_truth["Ethos"] = 1
                ground_truth["Ad Hominem"] = 1
                ground_truth["Logos"] = 1
                ground_truth["Distraction"] = 1
                ground_truth["Reasoning"] = 1

            if (label in ["Causal Oversimplification", "Black-and-white Fallacy/Dictatorship",
                          "Thought-terminating cliché"]):
                ground_truth[label] = 1
                ground_truth["Logos"] = 1
                ground_truth["Reasoning"] = 1
                ground_truth["Simplification"] = 1

        gt = list(ground_truth.keys())
        return gt

    def read_json_data(self, file_path):
        f = open(file_path, encoding="utf8")
        train_json_data = json.load(f)
        f.close()
        return train_json_data
