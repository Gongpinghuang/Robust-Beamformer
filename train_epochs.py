import argparse
from speechDataloader import SpeechData
import torch
from torch.utils import data
import torch.optim as optim
from net import CNN
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='./meta/train_Dataset.meta', type=str, help='Meta for audio')
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='./model/init.bin', type=str, help='Model to start training from')
parser.add_argument('--model_dst', default='./model/model_saved.bin', type=str, help='Model to save')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training samples')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers')
parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--scratch_directory', default=None, type=str, help='Directory to store temporary files')
args = parser.parse_args()

if __name__ == '__main__':

    # CUDA

    torch.backends.cudnn.enabled = False
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters

    params = {'batch_size': args.batch_size,
              'shuffle': args.shuffle,
              'num_workers': args.num_workers,
              'pin_memory': args.pin_memory}

    # Dataset

    dataset = SpeechData(data_meta=args.audio, file_json= args.json)
    # Dataloader

    dataloader = data.DataLoader(dataset, **params)

    # Model

    net = CNN(file_json=args.json).to(device)
    net.load_state_dict(torch.load(args.model_src))

    # Loss

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Training

    net.train()

    for epoch in range(0, args.num_epochs):
        for local_batch, local_labels in dataloader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward + optimize

            outputs = net(local_batch)

            loss = criterion(outputs*700, local_labels*700)
            loss.backward()
            optimizer.step()

            print(epoch, loss)

    # Save
    torch.save(net.state_dict(), args.model_dst)