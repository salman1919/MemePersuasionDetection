import json

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# from format_checker.task1_3 import read_classes
from transformers import AutoTokenizer

with open('cfg/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Read labels from the text file
with open('output-classes-subtask1.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

# Create a label encoding dictionary
label_encoding = {label_str: i for i, label_str in enumerate(labels)}

print(label_encoding)


def hierarchical_label_encoding(label):
    hierarchy = {
        "Ethos": {
            "Ad Hominem": ["Doubt", "Name calling/Labeling", "Smears", "Reductio ad hitlerum",
                           "Whataboutism"],
            "Bandwagon": ["Slogan"],
            "Appeal to authority": ["Slogan"],
            "Glittering generalities (Virtue)": ["Slogan"],
            "Transfer": []
        },
        "Pathos": {
            "Exaggeration/Minimisation": [],
            "Loaded Language": [],
            "Flag-waving": ["Slogan"],
            "Appeal to fear/prejudice": ["Slogan"],
            "Transfer": []
        },
        "Logos": {
            "Reasoning": {
                "Distraction": ["Misrepresentation of Someone's Position (Straw Man)",
                                "Presenting Irrelevant Data (Red Herring)", "Whataboutism"],
                "Simplification": ["Causal Oversimplification",
                                   "Black-and-white Fallacy/Dictatorship",
                                   "Thought-terminating cliché"]
            },
            "Justification": ["Bandwagon", "Appeal to authority", "Flag-waving",
                              "Appeal to fear/prejudice", "Slogan"],
            "Repetition": [],
            "Obfuscation, Intentional vagueness, Confusion": []
        }
    }

    # encoded_labels = []
    #
    # def encode(label, level):
    #     if label in hierarchy.keys():
    #         encoded_labels.append(label_encoding[label])
    #     else:
    #
    #
    # encode(hierarchy,label, 1)

    return label_encoding[label]


def task1_dataset():
    with open('dataset/annotations_v2/semeval2024_dev_release/subtask1/train.json', 'r',
              encoding="utf8") as d:
        train_data = json.load(d)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['text-model']['pretrain'])

    # Lists to store tokenized text and labels
    tokenized_input_ids = []
    tokenized_attention_masks = []
    labels_list = []

    # Tokenize text and encode labels
    max_length = 0
    count = 1
    for meme in train_data:

        if meme['text'] is not None and len(meme['labels']) != 0:
            text = meme['text']

            # Tokenize text
            tokenized_text = tokenizer(text, padding='max_length', truncation=True,
                                       return_tensors='pt')
            tokenized_input_ids.append(tokenized_text['input_ids'])
            tokenized_attention_masks.append(tokenized_text['attention_mask'])

            # Track maximum sequence length
            max_length = max(max_length, tokenized_text['input_ids'].shape[1])

            # Encode labels
            encoded_meme_labels = []
            for label in meme['labels']:
                encoded_label = hierarchical_label_encoding(label)
                encoded_meme_labels.append(encoded_label)

            labels_list.append(encoded_meme_labels)

    # Pad input tensors to the same size
    for i in range(len(tokenized_input_ids)):
        padding_length = max_length - tokenized_input_ids[i].shape[1]
        tokenized_input_ids[i] = torch.nn.functional.pad(tokenized_input_ids[i],
                                                         (0, padding_length),
                                                         value=tokenizer.pad_token_id)
        tokenized_attention_masks[i] = torch.nn.functional.pad(tokenized_attention_masks[i],
                                                               (0, padding_length), value=0)

    # Convert lists to tensors
    input_ids = torch.cat(tokenized_input_ids, dim=0)
    attention_masks = torch.cat(tokenized_attention_masks, dim=0)

    #
    # # Split the dataset into training and validation sets
    # input_ids_train, input_ids_val, attention_masks_train, attention_masks_val, labels_train, labels_val = train_test_split(
    #     input_ids, attention_masks, labels, test_size=0.2, random_state=42
    # )
    #
    # # Print the shapes of the training and validation sets print("Training data shape:",
    # input_ids_train.shape, attention_masks_train.shape, len(labels_train)) print("Validation
    # data shape:", input_ids_val.shape, attention_masks_val.shape, len(labels_val))

    # labels_tensor = encode_labels(labels_list, num_classes=28)

    # Split the dataset into training and validation sets
    input_ids_train, input_ids_val, attention_masks_train, attention_masks_val, labels_train, labels_val = train_test_split(
        input_ids, attention_masks, labels_list, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    input_ids_train = torch.tensor(input_ids_train)
    attention_masks_train = torch.tensor(attention_masks_train)

    labels_train = list(map(torch.as_tensor, labels_train))
    labels_train_tensor = torch.nested.as_nested_tensor(labels_train, dtype=torch.long)

    # labels_train = torch.Tensor(inner)


    input_ids_val = torch.tensor(input_ids_val)
    attention_masks_val = torch.tensor(attention_masks_val)

    labels_val = list(map(torch.as_tensor, labels_val))
    labels_val_tensor = torch.nested.as_nested_tensor(labels_val, dtype=torch.long)


    # Create DataLoader for training and validation sets
    batch_size = 16

    train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate)

    val_data = TensorDataset(input_ids_val, attention_masks_val, labels_val_tensor)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate)

    return train_loader, val_loader


def encode_labels(labels, num_classes):
    encoded_labels = []
    for example_labels in labels:
        encoded_labels_example = torch.zeros(num_classes)
        for label in example_labels:
            label_index = label_encoding[label]  # Convert label to index
            encoded_labels_example[label_index] = 1
        encoded_labels.append(encoded_labels_example)
    return encoded_labels


def custom_collate(batch):
    # Get the maximum length in the batch
    max_length = max(len(example[0]) for example in
                     batch)  # Assuming all tensors are of the same length along dim=0

    # Pad each tensor in the tuple to the maximum length
    padded_batch = []
    for example in batch:
        padded_example = []
        for tensor in example:
            # Pad tensor to the maximum length
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)))
            padded_example.append(padded_tensor)
        padded_batch.append(padded_example)

    # Stack padded examples
    return [torch.stack(example, dim=0) for example in zip(*padded_batch)]


# def custom_collate(batch):
#     # Pad sequences to the same length
#     max_len = max(len(labels) for labels in batch)
#     padded_labels = [torch.tensor(labels + [0] * (max_len - len(labels))) for labels in batch]
#     return padded_labels


if __name__ == '__main__':
    task1_dataset()
