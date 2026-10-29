import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import os

try:
    import medmnist
    from medmnist import INFO, Evaluator
    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False


class SyntheticMedicalDataset(Dataset):
    """
    Synthetic Medical Dataset fallback (simulating 28x28 chest X-ray images)
    used when medmnist dataset download is not available or offline.
    """
    def __init__(self, num_samples: int = 1000, num_classes: int = 2, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Generate synthetic 28x28 grayscale images (1x28x28)
        self.data = torch.randn(num_samples, 1, 28, 28)
        # Generate binary targets (0: Normal, 1: Pneumonia) with slight signal
        feature_sum = self.data.view(num_samples, -1).mean(dim=1)
        prob = torch.sigmoid(feature_sum * 2.0)
        self.targets = (prob > 0.5).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_medical_dataset(data_flag: str = "pneumoniamnist", data_dir: str = "./data"):
    """
    Loads MedMNIST dataset (e.g. PneumoniaMNIST).
    Falls back to SyntheticMedicalDataset if medmnist is unavailable.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if HAS_MEDMNIST:
        try:
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            train_dataset = DataClass(split='train', transform=data_transform, download=True, root=data_dir)
            val_dataset = DataClass(split='val', transform=data_transform, download=True, root=data_dir)
            test_dataset = DataClass(split='test', transform=data_transform, download=True, root=data_dir)
            
            # Squeeze targets if 2D
            train_dataset.labels = train_dataset.labels.squeeze()
            val_dataset.labels = val_dataset.labels.squeeze()
            test_dataset.labels = test_dataset.labels.squeeze()
            
            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            print(f"[Warning] Failed to load MedMNIST '{data_flag}': {e}. Using synthetic dataset.")
    
    # Fallback to synthetic medical dataset
    train_ds = SyntheticMedicalDataset(num_samples=2000, seed=42)
    val_ds = SyntheticMedicalDataset(num_samples=400, seed=43)
    test_ds = SyntheticMedicalDataset(num_samples=500, seed=44)
    return train_ds, val_ds, test_ds


def partition_data_non_iid(dataset, num_clients: int = 5, alpha: float = 0.5, seed: int = 42):
    """
    Partitions a PyTorch dataset across `num_clients` in a non-IID fashion
    using a Dirichlet distribution over class labels.
    
    alpha: Dirichlet concentration parameter (smaller alpha = more non-IID).
    """
    np.random.seed(seed)
    
    # Extract labels
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize proportions to match total samples in class c
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_split = np.split(idx_c, proportions)
        for i in range(num_clients):
            client_indices[i].extend(client_split[i])
            
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    return client_datasets, client_indices


def get_client_dataloaders(num_clients: int = 5, alpha: float = 0.5, batch_size: int = 32, return_client_sizes: bool = False):
    """
    Constructs DataLoaders for N simulated hospital clients and server test set.
    Returns: (client_train_loaders, client_val_loaders, test_loader, client_indices)
    or (client_train_loaders, client_val_loaders, test_loader, client_indices, client_sizes) if return_client_sizes=True.
    """
    train_ds, val_ds, test_ds = load_medical_dataset()
    
    client_subsets, client_indices = partition_data_non_iid(train_ds, num_clients=num_clients, alpha=alpha)
    
    client_train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in client_subsets
    ]
    
    client_sizes = [len(subset) for subset in client_subsets]
    
    # Validation loaders for clients
    client_val_subsets, _ = partition_data_non_iid(val_ds, num_clients=num_clients, alpha=alpha, seed=43)
    client_val_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for subset in client_val_subsets
    ]
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    if return_client_sizes:
        return client_train_loaders, client_val_loaders, test_loader, client_indices, client_sizes
    
    return client_train_loaders, client_val_loaders, test_loader, client_indices
