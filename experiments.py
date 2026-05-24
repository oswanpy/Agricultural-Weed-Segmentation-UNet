import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SugarBeetsTileDataset
from model import UNet
from engine import train_and_evaluate

BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_TILES = 2000

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 10

    data_dir = SCRIPT_DIR.parent / 'data'
    full_train_dataset = SugarBeetsTileDataset(
        root_dir=str(data_dir),
        split_file='train.txt',
        tile_size=256,
        max_samples=MAX_TILES,
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    class_weights = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Experiment 1: Varying Learning Rate
    for lr in [1e-3, 1e-4, 1e-5]:
        print(f"\n--- Experiment 1: Adam Optimizer | LR: {lr} ---")
        model = UNet(base_c=64).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        acc, iou, f1 = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)
        print(f"Result LR {lr} -> Acc: {acc:.4f} | mIoU: {iou:.4f} | mF1: {f1:.4f}")

    # Experiment 2: Varying Optimizer
    for opt_name in ['Adam', 'SGD']:
        print(f"\n--- Experiment 2: Constant LR: 1e-4 | Optimizer: {opt_name} ---")
        model = UNet(base_c=64).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-4) if opt_name == 'Adam' else optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        acc, iou, f1 = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)
        print(f"Result {opt_name} -> Acc: {acc:.4f} | mIoU: {iou:.4f} | mF1: {f1:.4f}")

    # Experiment 3: Varying Architecture Capacity
    for base_c in [64, 32]:
        print(f"\n--- Experiment 3: Constant LR & Adam | Base Channels: {base_c} ---")
        model = UNet(base_c=base_c).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        acc, iou, f1 = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)
        print(f"Result {base_c} Ch -> Acc: {acc:.4f} | mIoU: {iou:.4f} | mF1: {f1:.4f}")

if __name__ == "__main__":
    main()