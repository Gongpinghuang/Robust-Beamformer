import argparse
import torch

from net import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON with parameters')
parser.add_argument('--model_dst', default='./model/init.bin', type=str, help='Model to save')
args = parser.parse_args()

# Save

torch.save(CNN(file_json=args.json).state_dict(), args.model_dst)
