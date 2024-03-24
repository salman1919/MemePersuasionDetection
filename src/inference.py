import json

import numpy as np
import torch

from config import val_json_path, val_images_folder_path, device
from pre_process import ProcessDataset
from train import TextImageProcessor

val_dataset = ProcessDataset(val_json_path, val_images_folder_path, device, mode="val")

TI_P = TextImageProcessor(device)
TI_P.load_state_dict(torch.load("./model_v3.pt", map_location=torch.device(device)))
TI_P.eval()

print(TI_P)

validation_results = []
# for x in range(0,len(val_dataset)):
for x in range(0, 5):
    try:
        i_f, t_f, gt, id = val_dataset[x]
        output = TI_P(i_f, t_f)
        result = (output.detach().cpu().numpy() >= 0.5).astype(int)
        all_indexes = np.where(result == 1)[0]
        targets = []
        for i in all_indexes:
            targets.append(val_dataset.all_targets[i])
        validation_results.append({"id": id, "labels": targets})
    except Exception as e:
        print(e)
        print(x)

json_strings = [json.dumps(obj) for obj in validation_results]

# Write the JSON strings to a text file
with open('output.txt', 'w') as file:
    file.write('[\n')
    file.write(',\n'.join(json_strings))
    file.write('\n]')
