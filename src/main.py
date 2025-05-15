import argparse

import torch
from torchvision import models
import heatmap as hm
import utils as ut

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--models', type=str, nargs='+', choices=['mobilenet', 'resnet', 'clip'], help='list of models to use')
    parser.add_argument('--heatmap', type=str, choices=['gradcam', 'knn', 'zero_zone'], help='type of heatmap to create')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'fashionmnist', 'cifar100'], help='dataset to use')
    parser.add_argument('--save_path', type=str, help='path to save the heatmap')

    args = parser.parse_args()
    print(f"Creating {args.heatmap} heatmap for {args.dataset} dataset and saving to {args.save_path}")

    dataset = ut.load_dataset(args.dataset)
    heatmap = hm.create_heatmap(args.heatmap, dataset)
    

if __name__ == "__main__":
    main()