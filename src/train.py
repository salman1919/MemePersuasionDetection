import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import train_json_path, train_images_folder_path, device, batch_size, epochs
from pre_process import ProcessDataset


class TextProcessor(nn.Module):
    def __init__(self, device):
        super(TextProcessor, self).__init__()
        self.fc1 = nn.Linear(768, 1000, device=device)
        self.fc2 = nn.Linear(1000, 500, device=device)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = F.relu(self.fc2(x))
        return x


class ImageProcessor(nn.Module):
    def __init__(self, device):
        super(ImageProcessor, self).__init__()
        self.fc1 = nn.Linear(1000, 1000, device=device)
        self.fc2 = nn.Linear(1000, 500, device=device)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = F.relu(self.fc2(x))
        return x


class TextImageProcessor(nn.Module):
    def __init__(self, device):
        super(TextImageProcessor, self).__init__()
        self.text_processor = TextProcessor(device)
        self.image_processor = ImageProcessor(device)
        self.fc1 = nn.Linear(1000, 500, device=device)
        self.fc2 = nn.Linear(500, 250, device=device)
        self.fc3 = nn.Linear(250, 250, device=device)
        self.fc4 = nn.Linear(250, 30, device=device)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, image_feature, text_feature):
        i_f = self.image_processor(image_feature)
        t_f = self.text_processor(text_feature)
        c_f = torch.concat((i_f, t_f))
        x = F.relu(self.fc1(c_f))
        x = F.relu(self.fc2(x))
        x = self.dp(x)
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


if __name__ == "__main__":

    train_dataset = ProcessDataset(train_json_path, train_images_folder_path, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"This is the total number of instances in train dataset : {len(train_dataset)}")

    TI_P = TextImageProcessor(device)
    print(TI_P)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(TI_P.parameters(), lr=0.001)

    TI_P.train()

    for epoch in range(epochs):
        running_loss = 0.0
        try:
            image_batch, text_batch, target_batch = next(iter(train_dataloader))
        except Exception as e:
            print(e)
            print("Missed an epoch")
            continue
        for input_image, input_text, target in zip(image_batch, text_batch, target_batch):
            try:
                # i_f , t_f , gt = train_dataset[x]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = TI_P(input_image, input_text)
                # print(output)
                # print(target)
                loss = criterion(output.float().to("cpu"), target.float().to("cpu"))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

                # if x % 10 == 9:
                #     print(f"[{epoch + 1}, {x + 1}] loss: {running_loss / 10:.3f}")
                #     running_loss = 0.0
            except:
                pass

        print(f"Epoch # {epoch} has average running loss = {running_loss / batch_size}")
        if epoch % 20 == 0:
            torch.save(TI_P.state_dict(), "./model_v1.pt")

    print("Finished Training")
