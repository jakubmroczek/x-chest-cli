import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

NET_PATH = './net.pth'

class ProgramArguments:
    def __init__(self) -> None:
        self.net_path = NET_PATH
        self.images_path = None

def main(args):
    net = _load_net(args.net_path)
    dataset = _load_dataset(args.images_path)
    # no shuffle cause we want to track files
    data_loader = DataLoader(dataset, batch_size = 1, shuffle=False)
    result = _predict(net, data_loader)
    _log(result, dataset)

def _load_net(path):
    device = torch.device('cpu')
    model =  models.squeezenet1_0(pretrained=False)
    num_classes = 2
    # We fine tuned slightly the net
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))  
    model.load_state_dict(torch.load(NET_PATH, map_location=device))
    model.eval()
    return model

def _load_dataset(path):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset = datasets.ImageFolder(path, transform=transform)
    return dataset

def _predict(model, data_loader):
    device = torch.device('cpu')
    running_correct = 0.0
    running_total = 0.0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        # Image is really a datalaoder with 1 image
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.append(labels.item())
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_labels.append(preds.item())
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()
        
    return (true_labels, pred_labels)

def _log(result, dataset):
    index = 1
    image_paths = dataset.imgs
    for true_label, pred_label, path in zip(result[0], result[1], image_paths):
        print(f'\t{index}. label: {true_label} predicted: {pred_label} path: {path}', end='')
        if true_label != pred_label:
            print(' MISSED')
        else:
            print('')
        index += 1

def _make_cli_parser():
    parser = argparse.ArgumentParser(description='Tool to interact with learnend net')
    parser.add_argument('--images', required=True)
    return parser

if __name__ == '__main__':
    args = _make_cli_parser().parse_args()
    program_args = ProgramArguments()
    program_args.images_path = args.images
    main(program_args)