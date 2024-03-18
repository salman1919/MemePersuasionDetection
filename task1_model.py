import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
import numpy as np

from dataset import task1_dataset


# Define your model class with RoBERTa
class RoBERTaForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(RoBERTaForMultiLabelClassification, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


if __name__ == '__main__':
    train_loader, val_loader = task1_dataset()

    # Instantiate your model
    num_labels = 22  # Update this with the number of labels in your dataset
    model = RoBERTaForMultiLabelClassification(num_labels)

    # Define your loss function
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Define number of epochs
    epochs = 10

    # Training loop
    for epoch in range(epochs):
        print("Epoch: "+ str(epoch))
        model.train()
        total_loss = 0

        # Iterate over training batches
        for batch in train_loader:
            input_ids, attention_masks, labels = batch
            print(labels.size())

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_masks)

            # Compute loss
            loss = criterion(logits, labels.float())  # Convert labels to float for BCEWithLogitsLoss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0

            # Iterate over validation batches
            for batch in val_loader:
                input_ids, attention_masks, labels = batch

                # Forward pass
                logits = model(input_ids, attention_masks)

                # Compute predictions
                predictions = (
                            torch.sigmoid(logits) > 0.5).long()  # Convert logits to binary predictions

                # Compute accuracy
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate validation accuracy
            val_accuracy = total_correct / total_samples
            print(f'Validation Accuracy: {val_accuracy:.4f}')

    torch.save(model.state_dict(), 'roberta_multilabel_classification_model.pth')
