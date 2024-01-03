import torch
import torchvision.datasets as datasets
import torchvision.transforms as TF
from torch.utils.data import DataLoader


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_dataset(dataset_name="MNIST", root_dir="data"):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(
                (32, 32), interpolation=TF.InterpolationMode.BICUBIC, antialias=True
            ),
            # TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(
            root=root_dir, train=True, download=True, transform=transforms
        )
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transforms
        )
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transforms
        )
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root=root_dir, transform=transforms)

    return dataset


def get_dataloader(
    dataset_name="MNIST",
    batch_size=32,
    pin_memory=False,
    shuffle=True,
    num_workers=0,
    device="cpu",
    root_dir="data",
):
    dataset = get_dataset(dataset_name=dataset_name, root_dir=root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader
