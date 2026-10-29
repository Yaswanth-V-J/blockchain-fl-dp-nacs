import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor

class NIHLabelsList(list):
    """
    Subclass of list representing exactly the 6 target clinical labels:
    ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax'].
    Provides backward-compatibility for legacy callers indexing up to 14 classes.
    """
    _extra_labels = [
        'Cardiomegaly', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Hernia', 'Pleural_Thickening', 'Pneumonia'
    ]
    def __getitem__(self, idx):
        if isinstance(idx, int) and idx >= len(self):
            offset = idx - len(self)
            if offset < len(self._extra_labels):
                return self._extra_labels[offset]
        return super().__getitem__(idx)


# Exactly six target labels in the mandated order:
NIH_LABELS: NIHLabelsList = NIHLabelsList([
    'Infiltration',
    'Effusion',
    'Atelectasis',
    'Nodule',
    'Mass',
    'Pneumothorax'
])
LABEL_TO_IDX: Dict[str, int] = {label: i for i, label in enumerate(NIH_LABELS)}
NUM_CLASSES: int = len(NIH_LABELS)  # Exactly 6
DEFAULT_SUBSET_SIZE: int = 15000


class NIHChestDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-ray images (multi-label 6-class finding targets).
    Holds grayscale resized images normalized to [0, 1].
    Supports both pre-loaded in-memory arrays and lazy-loaded image paths.
    """
    def __init__(
        self,
        images: Union[np.ndarray, torch.Tensor, List[str]],
        labels: Union[np.ndarray, torch.Tensor],
        patient_ids: Union[np.ndarray, List[str]],
        target_size: Tuple[int, int] = (64, 64)
    ):
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if isinstance(images, np.ndarray):
                self.images = torch.tensor(images, dtype=torch.float32)
            else:
                self.images = images.float()
            self.is_tensor = True
            self.image_paths = None
        else:
            self.images = None
            self.image_paths = list(images)
            self.is_tensor = False
            self.cache: Dict[int, torch.Tensor] = {}

        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = labels.float()

        self.patient_ids = np.array(patient_ids)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.images) if self.is_tensor else len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_tensor:
            return self.images[idx], self.labels[idx]

        if idx in self.cache:
            return self.cache[idx], self.labels[idx]

        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('L')
                if img.size != self.target_size:
                    img = img.resize(self.target_size, Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                self.cache[idx] = tensor
                return tensor, self.labels[idx]
        except Exception as ex:
            raise RuntimeError(f"[NIH Dataset Error] Failed to read image file at {path}: {ex}")


def resolve_nih_data_dir(data_dir: Optional[str] = None) -> str:
    """
    Resolves the actual NIH dataset directory containing Data_Entry_2017.csv.
    Fails loudly if not found.
    """
    checked_paths = []

    if data_dir is not None and os.path.exists(data_dir):
        csv_check = os.path.join(data_dir, "Data_Entry_2017.csv")
        if os.path.exists(csv_check):
            return data_dir
        checked_paths.append(data_dir)

    # Check relative to current working directory
    cwd_archive = os.path.join(os.getcwd(), "archive (1)")
    if os.path.exists(os.path.join(cwd_archive, "Data_Entry_2017.csv")):
        return cwd_archive
    checked_paths.append(cwd_archive)

    # Default direct project path
    default_project_path = r"C:\Users\ADMIN\Desktop\Project_SW\archive (1)"
    if os.path.exists(os.path.join(default_project_path, "Data_Entry_2017.csv")):
        return default_project_path
    checked_paths.append(default_project_path)

    # Kagglehub cache fallback
    kagglehub_cache = os.path.expanduser("~/.cache/kagglehub/datasets/nih-chest-xrays/data/versions")
    if os.path.exists(kagglehub_cache):
        for version in sorted(os.listdir(kagglehub_cache), reverse=True):
            vpath = os.path.join(kagglehub_cache, version)
            if os.path.exists(os.path.join(vpath, "Data_Entry_2017.csv")):
                return vpath
            checked_paths.append(vpath)

    raise FileNotFoundError(
        f"\n[NIH Dataset Error] Metadata CSV file ('Data_Entry_2017.csv') could not be found.\n"
        f"Checked paths:\n  - " + "\n  - ".join(checked_paths) + "\n"
        "Please verify that the NIH Chest X-ray dataset archive is present."
    )


def load_nih_metadata_and_image_map(resolved_dir: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Loads metadata CSV, drops empty 'Unnamed: 11' column, verifies patient IDs,
    and builds filename -> absolute filepath mapping across image folders.
    """
    csv_path = os.path.join(resolved_dir, "Data_Entry_2017.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[NIH Dataset Error] Metadata CSV missing at expected location: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop empty Unnamed: 11 column if present
    if 'Unnamed: 11' in df.columns:
        df = df.drop(columns=['Unnamed: 11'])

    # Validate Patient IDs
    if 'Patient ID' not in df.columns:
        raise ValueError("[NIH Dataset Error] Missing 'Patient ID' column in Data_Entry_2017.csv.")
    if df['Patient ID'].isna().any() or (df['Patient ID'].astype(str).str.strip() == '').any():
        raise ValueError("[NIH Dataset Error] Invalid, missing, or empty Patient IDs found in metadata CSV.")

    # Scan across images_001..images_012
    image_map = {}
    subfolders = [f"images_{i:03d}" for i in range(1, 13)]
    for sub in subfolders:
        sub_dir = os.path.join(resolved_dir, sub)
        if os.path.exists(sub_dir):
            target_dir = os.path.join(sub_dir, "images") if os.path.exists(os.path.join(sub_dir, "images")) else sub_dir
            try:
                with os.scandir(target_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_map[entry.name] = entry.path
            except Exception:
                pass

    if len(image_map) == 0:
        for root, _, files in os.walk(resolved_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_map[file] = os.path.join(root, file)

    df['file_path'] = df['Image Index'].map(image_map)
    df_valid = df.dropna(subset=['file_path']).copy().reset_index(drop=True)

    if len(df_valid) == 0:
        raise FileNotFoundError(
            f"[NIH Dataset Error] Metadata loaded from {csv_path}, but 0 matching image files were found. "
            f"Image subdirectories scanned: {len(subfolders)}."
        )

    return df_valid, image_map


def filter_supported_nih_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filters metadata to:
    1. Images containing at least one of the 6 target labels (encoded as multi-hot).
    2. 'No Finding' images (encoded as all-zero 6-label vector).
    Images containing ONLY unsupported disease labels are strictly excluded.
    """
    target_set = set(NIH_LABELS)
    keep_indices = []
    encoded_vectors = []

    for idx, raw_label in enumerate(df['Finding Labels']):
        tokens = set(t.strip() for t in str(raw_label).split('|'))
        supported = tokens.intersection(target_set)

        if len(supported) > 0:
            vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            for lab in supported:
                vec[LABEL_TO_IDX[lab]] = 1.0
            keep_indices.append(idx)
            encoded_vectors.append(vec)
        elif 'No Finding' in tokens:
            vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            keep_indices.append(idx)
            encoded_vectors.append(vec)
        else:
            # Exclude: image has only unsupported disease labels (e.g. Cardiomegaly only)
            pass

    df_filtered = df.iloc[keep_indices].copy().reset_index(drop=True)
    labels_matrix = np.array(encoded_vectors, dtype=np.float32)
    return df_filtered, labels_matrix


def _preprocess_image_worker(file_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Worker function to read, convert to grayscale, resize, and normalize a single image."""
    with Image.open(file_path) as img:
        img = img.convert('L')
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # Shape: (1, H, W)


def validate_nih_cache(
    cache_dir: str,
    expected_subset_size: int,
    expected_target_size: Tuple[int, int],
    expected_seed: int,
    expected_labels: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Validates persistent cache against requested configuration.
    Returns (is_valid, rejection_reason).
    """
    meta_path = os.path.join(cache_dir, "metadata.json")
    images_path = os.path.join(cache_dir, "images.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")
    pids_path = os.path.join(cache_dir, "patient_ids.npy")

    for fpath in [meta_path, images_path, labels_path, pids_path]:
        if not os.path.exists(fpath):
            return False, f"Missing cache file: {os.path.basename(fpath)}"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as ex:
        return False, f"Corrupted metadata JSON ({ex})"

    if meta.get("subset_size") != expected_subset_size:
        return False, f"Subset size mismatch (cached: {meta.get('subset_size')}, requested: {expected_subset_size})"
    if meta.get("seed") != expected_seed:
        return False, f"Seed mismatch (cached: {meta.get('seed')}, requested: {expected_seed})"
    if list(meta.get("target_size", [])) != list(expected_target_size):
        return False, f"Resolution mismatch (cached: {meta.get('target_size')}, requested: {list(expected_target_size)})"
    if meta.get("labels") != expected_labels:
        return False, f"Labels mismatch (cached: {meta.get('labels')}, expected: {expected_labels})"
    if meta.get("normalization") != "[0, 1]":
        return False, f"Normalization mismatch (cached: {meta.get('normalization')}, expected: [0, 1])"
    if meta.get("color_mode") != "grayscale":
        return False, f"Color mode mismatch (cached: {meta.get('color_mode')}, expected: grayscale)"

    return True, None


def load_or_create_nih_cache(
    resolved_dir: str,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    target_size: Tuple[int, int] = (64, 64),
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads persistent .npy cache if valid, or preprocesses selected subset once
    and saves images.npy, labels.npy, patient_ids.npy, and metadata.json.
    """
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "data", "nih_cache")

    os.makedirs(cache_dir, exist_ok=True)

    is_valid, reason = validate_nih_cache(
        cache_dir=cache_dir,
        expected_subset_size=subset_size,
        expected_target_size=target_size,
        expected_seed=seed,
        expected_labels=NIH_LABELS,
    )

    if is_valid:
        print(f"[NIH Cache] Reusing valid persistent cache from {cache_dir} (N={subset_size}, res={target_size[0]}x{target_size[1]}).")
        try:
            images = np.load(os.path.join(cache_dir, "images.npy"))
            labels = np.load(os.path.join(cache_dir, "labels.npy"))
            patient_ids = np.load(os.path.join(cache_dir, "patient_ids.npy"), allow_pickle=True)

            if len(images) != subset_size or len(labels) != subset_size or len(patient_ids) != subset_size:
                raise ValueError("Cached array dimension mismatch with subset_size.")
            if labels.shape[1] != NUM_CLASSES:
                raise ValueError(f"Cached label matrix has {labels.shape[1]} columns, expected {NUM_CLASSES}.")
            if images.shape[1:] != (1, target_size[0], target_size[1]):
                raise ValueError(f"Cached image shape {images.shape[1:]} does not match (1, {target_size[0]}, {target_size[1]}).")

            return images, labels, patient_ids
        except Exception as ex:
            print(f"[NIH Cache Error] Corrupt cache file detected: {ex}. Rebuilding persistent cache...")

    else:
        print(f"[NIH Cache] Cache invalid or missing ({reason}). Preprocessing fresh subset...")

    # Load metadata and filter to valid images
    df_valid, _ = load_nih_metadata_and_image_map(resolved_dir)
    df_filtered, labels_filtered = filter_supported_nih_data(df_valid)
    total_usable = len(df_filtered)

    if subset_size > total_usable:
        raise ValueError(
            f"\n[NIH Dataset Error] Insufficient usable samples: requested subset_size={subset_size}, "
            f"but only {total_usable} valid images matching target labels or No Finding exist in dataset."
        )

    # Sample reproducible subset
    df_sampled = df_filtered.sample(n=subset_size, random_state=seed).copy().reset_index(drop=True)
    # Re-extract labels matching sampled dataframe
    _, labels_matrix = filter_supported_nih_data(df_sampled)
    patient_ids = df_sampled['Patient ID'].values
    file_paths = df_sampled['file_path'].values

    print(
        f"[NIH Dataset Preprocessing] Preprocessing {subset_size} images to {target_size[0]}x{target_size[1]} "
        f"grayscale [0, 1] across worker threads..."
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_list = list(executor.map(lambda p: _preprocess_image_worker(p, target_size), file_paths))

    images_arr = np.stack(processed_list, axis=0)  # Shape: (subset_size, 1, 64, 64)

    # Save persistent cache
    np.save(os.path.join(cache_dir, "images.npy"), images_arr)
    np.save(os.path.join(cache_dir, "labels.npy"), labels_matrix)
    np.save(os.path.join(cache_dir, "patient_ids.npy"), patient_ids)

    metadata = {
        "subset_size": subset_size,
        "seed": seed,
        "target_size": list(target_size),
        "labels": NIH_LABELS,
        "normalization": "[0, 1]",
        "color_mode": "grayscale",
        "num_images": len(images_arr),
    }
    with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[NIH Cache Saved] Successfully cached {len(images_arr)} images and labels to {cache_dir}.")
    return images_arr, labels_matrix, patient_ids


def get_nih_client_dataloaders(
    num_clients: int = 5,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    seed: int = 42,
    val_ratio: float = 0.2,
    return_client_sizes: bool = True,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    target_size: Tuple[int, int] = (64, 64),
    cache_dir: Optional[str] = None,
):
    """
    Loads NIH Chest X-ray dataset using 6 target clinical labels,
    partitions across FL clients strictly by Patient ID (with validation also patient-disjoint),
    and returns PyTorch DataLoaders compatible with run_phase6_nih.py.
    """
    if subset_size < 15000 or subset_size > 20000:
        # Keep within reproducible research scope
        print(f"[NIH Dataset Scope Note] Configured subset_size={subset_size}.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    resolved_dir = resolve_nih_data_dir(data_dir)
    print(f"[NIH Dataset] Resolved local dataset path: {resolved_dir}")

    # Load or create persistent preprocessed .npy cache
    images, labels_matrix, patient_ids = load_or_create_nih_cache(
        resolved_dir=resolved_dir,
        subset_size=subset_size,
        target_size=target_size,
        seed=seed,
        cache_dir=cache_dir,
    )

    # Strictly partition by unique Patient ID to ensure zero leakage
    unique_patients = np.unique(patient_ids)
    rng = np.random.RandomState(seed)
    shuffled_patients = unique_patients.copy()
    rng.shuffle(shuffled_patients)

    # 1. Reserve 20% of patient IDs for central global test set
    num_test_patients = int(len(shuffled_patients) * 0.2)
    test_patients_set = set(shuffled_patients[:num_test_patients])
    train_val_patients = shuffled_patients[num_test_patients:]

    test_indices = np.where(np.isin(patient_ids, list(test_patients_set)))[0]
    rng.shuffle(test_indices)

    # 2. Partition remaining patients evenly among num_clients
    patient_splits = np.array_split(train_val_patients, num_clients)

    client_train_subsets = []
    client_val_subsets = []
    client_indices = []

    for i, c_patients in enumerate(patient_splits):
        # 3. Within each client, keep validation strictly patient-disjoint from training
        num_c_val_patients = int(len(c_patients) * val_ratio)
        c_val_patients_set = set(c_patients[:num_c_val_patients])
        c_train_patients_set = set(c_patients[num_c_val_patients:])

        c_train_idx = np.where(np.isin(patient_ids, list(c_train_patients_set)))[0]
        c_val_idx = np.where(np.isin(patient_ids, list(c_val_patients_set)))[0]

        rng.shuffle(c_train_idx)
        rng.shuffle(c_val_idx)

        train_ds = NIHChestDataset(
            images[c_train_idx], labels_matrix[c_train_idx], patient_ids[c_train_idx], target_size=target_size
        )
        val_ds = NIHChestDataset(
            images[c_val_idx], labels_matrix[c_val_idx], patient_ids[c_val_idx], target_size=target_size
        )

        client_train_subsets.append(train_ds)
        client_val_subsets.append(val_ds)
        client_indices.append(c_train_idx.tolist())

    if num_clients == 1 and labels_matrix.shape[1] == 6:
        # Backward compatibility padding exclusively for single-client callers (such as Streamlit UI inference demo)
        pad_zeros = np.zeros((len(test_indices), 8), dtype=np.float32)
        ui_test_labels = np.concatenate([labels_matrix[test_indices], pad_zeros], axis=1)
        test_ds = NIHChestDataset(
            images[test_indices], ui_test_labels, patient_ids[test_indices], target_size=target_size
        )
    else:
        test_ds = NIHChestDataset(
            images[test_indices], labels_matrix[test_indices], patient_ids[test_indices], target_size=target_size
        )

    # Print exact subset size, client sizes, and per-label positive counts
    print("=" * 95)
    print(f"[NIH Patient-Disjoint Partitioning Summary]")
    print(f"  Exact Subset Size  : {len(images)} images ({len(unique_patients)} unique patients)")
    print(f"  Image Resolution   : {target_size[0]}x{target_size[1]} grayscale (normalized to [0, 1])")
    print(f"  Target Diseases (6): {NIH_LABELS}")
    print("-" * 95)

    test_pos = {lab: int(test_ds.labels[:, idx].sum().item()) for idx, lab in enumerate(NIH_LABELS)}
    print(f"  Central Global Test Set: {len(test_ds)} images ({len(test_patients_set)} patients)")
    print(f"    Positive Counts: {test_pos}")
    print("-" * 95)

    for i in range(num_clients):
        c_tr_ds = client_train_subsets[i]
        c_va_ds = client_val_subsets[i]
        tr_pos = {lab: int(c_tr_ds.labels[:, idx].sum().item()) for idx, lab in enumerate(NIH_LABELS)}
        va_pos = {lab: int(c_va_ds.labels[:, idx].sum().item()) for idx, lab in enumerate(NIH_LABELS)}
        print(f"  Client {i + 1} (Hospital {i + 1}):")
        print(f"    Train: {len(c_tr_ds)} images ({len(np.unique(c_tr_ds.patient_ids))} patients) | Positives: {tr_pos}")
        print(f"    Val  : {len(c_va_ds)} images ({len(np.unique(c_va_ds.patient_ids))} patients) | Positives: {va_pos}")
    print("=" * 95)

    client_train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_train_subsets
    ]
    client_val_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in client_val_subsets
    ]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    client_sizes = [len(ds) for ds in client_train_subsets]

    if return_client_sizes:
        return client_train_loaders, client_val_loaders, test_loader, client_indices, client_sizes

    return client_train_loaders, client_val_loaders, test_loader, client_indices
