import argparse
import numpy as np

from speechDataloader import SpeechData
from net import CNN

import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='./meta/AUDIO_META.meta', type=str, help='Meta for audio')
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='./model/model_final.bin', type=str, help='Model to evaluate from')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training samples')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers')
parser.add_argument('--scratch_directory', default=None, type=str, help='Directory to store temporary files')
args = parser.parse_args()

# CUDA
if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters

    params = {'batch_size': args.batch_size,
              'shuffle': args.shuffle,
              'num_workers': args.num_workers}

    # Dataset

    dataset = SpeechData(data_meta=args.audio, file_json= args.json)

    # Dataloader

    dataloader = data.DataLoader(dataset, **params)

    # Model

    net = CNN(file_json=args.json).to(device)
    net.load_state_dict(torch.load(args.model_src))

    # Loss

    criterion = nn.MSELoss()

    # Evaluate

    nBatches = len(dataloader)
    total_loss = 0.0

    net.eval()

    for local_batch, local_labels in dataloader:
        # Transfer to GPU

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Forward

        outputs = net(local_batch)

        loss = criterion(outputs*700, local_labels*700)
        total_loss += loss.item()

        print(total_loss)

    mean_loss = total_loss / (nBatches * args.batch_size)

    print(mean_loss)
