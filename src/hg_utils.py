from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import default_data_collator
import torch

def load_hf_dataset(dataset_name: str, train: bool, batch_size: int = 64):
    """
    Loads a dataset using the Hugging Face 'datasets' library.
    Now correctly maps local names to Hub names.
    """
    hf_name_map = {
        'tiny-imagenet': 'zh-plus/tiny-imagenet'
    }

    hub_name = hf_name_map.get(dataset_name)
    if not hub_name:
        raise ValueError(f"Dataset '{dataset_name}' does not have a defined Hugging Face Hub name in hg_utils.py.")

    print(f"Loading '{hub_name}' from Hugging Face (Train={train})...")

    full_dataset = load_dataset(hub_name)

    split_name = 'train' if train else 'valid'
    dataset = full_dataset[split_name]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def custom_collate_fn(features):
        """
        Manually stacks the tensors for images and labels into a batch.
        'features' is a list of dictionaries: [{'pixel_values': tensor, 'label': int}, ...]
        """
        # Stack all the image tensors into a single tensor
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        # Create a tensor from the list of integer labels
        labels = torch.tensor([f['label'] for f in features])
        # Return the desired (images, labels) tuple
        return pixel_values, labels
    
    def apply_transforms(examples):
        examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
        del examples['image']
        return examples

    dataset.set_transform(apply_transforms)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return dataset, data_loader