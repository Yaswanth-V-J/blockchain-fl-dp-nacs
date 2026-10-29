import os
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

class HARDataset(Dataset):
    """
    PyTorch Dataset for UCI Human Activity Recognition (UCI HAR) tabular data.
    Features: 561 pre-extracted sensor features per sample.
    Targets: 6 activity classes (0: Walking, 1: Walking Upstairs, 2: Walking Downstairs, 3: Sitting, 4: Standing, 5: Laying).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray = None):
        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)
        self.subjects = np.array(subjects) if subjects is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SyntheticHARDataset(Dataset):
    """
    Synthetic UCI HAR fallback dataset (561 tabular features, 6 activity classes, 5 simulated subjects).
    Constructed with strong class signal so models can learn cleanly if offline.
    """
    def __init__(self, num_samples: int = 1500, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 6 activity classes (0 to 5)
        self.targets = torch.tensor(np.random.randint(0, 6, size=num_samples), dtype=torch.long)
        # 561 continuous features with class signal in first 30 features
        data_np = np.random.randn(num_samples, 561).astype(np.float32)
        for c in range(6):
            c_mask = (self.targets.numpy() == c)
            data_np[c_mask, c * 5 : (c + 1) * 5] += 2.5  # Strong discriminative feature signal
            
        self.data = torch.tensor(data_np, dtype=torch.float32)
        # 5 simulated subject IDs (1 to 5)
        self.subjects = np.random.choice([1, 2, 3, 4, 5], size=num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def download_and_extract_uci_har(data_dir: str = "./data") -> str:
    """
    Downloads and extracts the UCI HAR dataset zip file if not present.
    Returns the path to the unzipped root directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    har_dir = os.path.join(data_dir, "UCI HAR Dataset")

    if os.path.exists(har_dir) and os.path.exists(os.path.join(har_dir, "train")):
        return har_dir

    # Check for existing zip files in data_dir
    possible_zips = [
        os.path.join(data_dir, "UCI HAR Dataset.zip"),
        os.path.join(data_dir, "uci_har.zip"),
    ]

    for zip_path in possible_zips:
        if os.path.exists(zip_path):
            print(f"[HAR Dataset] Extracting existing zip file {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                if os.path.exists(har_dir) and os.path.exists(os.path.join(har_dir, "train")):
                    return har_dir
            except Exception as e:
                print(f"[HAR Dataset Warning] Extraction failed for {zip_path}: {e}")

    # Attempt download if zip does not exist
    download_zip_path = os.path.join(data_dir, "uci_har.zip")
    print(f"[HAR Dataset] Downloading UCI HAR dataset from {UCI_HAR_URL}...")
    try:
        req = urllib.request.Request(UCI_HAR_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(download_zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[HAR Dataset] Extracting {download_zip_path}...")
        with zipfile.ZipFile(download_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        if os.path.exists(har_dir) and os.path.exists(os.path.join(har_dir, "train")):
            return har_dir
    except Exception as e:
        print(f"[HAR Dataset Warning] Download/Extraction failed: {e}. Falling back to synthetic HAR dataset.")

    return ""


def load_raw_har_split(split_dir: str, prefix: str):
    """
    Loads raw X, y, and subject arrays from a UCI HAR split directory ('train' or 'test').
    """
    x_file = os.path.join(split_dir, f"X_{prefix}.txt")
    y_file = os.path.join(split_dir, f"y_{prefix}.txt")
    sub_file = os.path.join(split_dir, f"subject_{prefix}.txt")

    if not (os.path.exists(x_file) and os.path.exists(y_file)):
        raise FileNotFoundError(f"Missing HAR files in {split_dir}")

    X = np.loadtxt(x_file)
    y_raw = np.loadtxt(y_file, dtype=int)
    # Ensure 0-indexed targets (0 to 5)
    if np.min(y_raw) >= 1:
        y = y_raw - 1
    else:
        y = y_raw

    subjects = np.loadtxt(sub_file, dtype=int) if os.path.exists(sub_file) else None
    return X, y, subjects


def load_uci_har_dataset(data_dir: str = "./data"):
    """
    Loads the real UCI HAR dataset if available (with training set feature standardization),
    or synthetic fallback dataset.
    Returns (train_dataset, test_dataset).
    """
    har_root = download_and_extract_uci_har(data_dir)

    if har_root and os.path.exists(os.path.join(har_root, "train")):
        try:
            train_dir = os.path.join(har_root, "train")
            test_dir = os.path.join(har_root, "test")

            X_train, y_train, sub_train = load_raw_har_split(train_dir, "train")
            X_test, y_test, sub_test = load_raw_har_split(test_dir, "test")

            # Feature standardization (fit on training data only)
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

            train_ds = HARDataset(X_train, y_train, sub_train)
            test_ds = HARDataset(X_test, y_test, sub_test)
            print(f"[HAR Dataset] Real UCI HAR loaded successfully ({len(train_ds)} train, {len(test_ds)} test samples).")
            return train_ds, test_ds
        except Exception as e:
            print(f"[HAR Dataset Warning] Failed to parse raw UCI HAR files: {e}. Using synthetic dataset.")

    # Fallback synthetic dataset
    train_ds = SyntheticHARDataset(num_samples=2500, seed=42)
    test_ds = SyntheticHARDataset(num_samples=500, seed=43)
    print("[HAR Dataset] Using Synthetic HAR Dataset fallback.")
    return train_ds, test_ds


def get_har_client_dataloaders(num_clients: int = 5, batch_size: int = 32, val_ratio: float = 0.2, seed: int = 42, data_dir: str = "./data", return_client_sizes: bool = False):
    """
    Constructs DataLoaders for N simulated clients (partitioned by REAL subject ID)
    and a global server test set.

    Interface matches dataset.py's get_client_dataloaders():
    Returns: (client_train_loaders, client_val_loaders, test_loader, client_indices)
    or (client_train_loaders, client_val_loaders, test_loader, client_indices, client_sizes) if return_client_sizes=True.
    """
    train_ds, test_ds = load_uci_har_dataset(data_dir=data_dir)

    np.random.seed(seed)

    if hasattr(train_ds, 'subjects') and train_ds.subjects is not None:
        unique_subjects = np.unique(train_ds.subjects)
        if len(unique_subjects) >= num_clients:
            chosen_subjects = unique_subjects[:num_clients]
        else:
            chosen_subjects = unique_subjects
            
        client_train_subsets = []
        client_val_subsets = []
        client_indices = []

        for i, subj_id in enumerate(chosen_subjects):
            subj_indices = np.where(train_ds.subjects == subj_id)[0]
            np.random.shuffle(subj_indices)

            val_size = int(len(subj_indices) * val_ratio)
            train_idx = subj_indices[val_size:]
            val_idx = subj_indices[:val_size]

            client_train_subsets.append(Subset(train_ds, train_idx))
            client_val_subsets.append(Subset(train_ds, val_idx))
            client_indices.append(train_idx.tolist())

        while len(client_train_subsets) < num_clients:
            idx = len(client_train_subsets) % len(chosen_subjects)
            client_train_subsets.append(client_train_subsets[idx])
            client_val_subsets.append(client_val_subsets[idx])
            client_indices.append(client_indices[idx])
    else:
        total_len = len(train_ds)
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)

        client_train_subsets = []
        client_val_subsets = []
        client_indices = []

        for split in splits:
            val_size = int(len(split) * val_ratio)
            client_train_subsets.append(Subset(train_ds, split[val_size:]))
            client_val_subsets.append(Subset(train_ds, split[:val_size]))
            client_indices.append(split[val_size:].tolist())

    client_train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in client_train_subsets
    ]

    client_sizes = [len(subset) for subset in client_train_subsets]

    client_val_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False)
        for subset in client_val_subsets
    ]

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if return_client_sizes:
        return client_train_loaders, client_val_loaders, test_loader, client_indices, client_sizes

    return client_train_loaders, client_val_loaders, test_loader, client_indices
