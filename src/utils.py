import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_name: str) -> DataLoader:
    """
    Load a dataset from torchvision.
    
    Args:
        dataset_name (str): Name of the dataset to load. Options are 'imagenet', 'fashionmnist', 'cifar100'.
    
    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    
    if dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageNet(root='data/imagenet', split='val', transform=transform)

    elif dataset_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.FashionMNIST(root='data/fashionmnist', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR100(root='data/cifar100', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return DataLoader(dataset, batch_size=32, shuffle=False)
