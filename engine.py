import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    print(f"Starting training for {epochs} epochs")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Batch size: {train_loader.batch_size}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", unit="batch") if tqdm else train_loader
        for batch_idx, (imgs, masks) in enumerate(train_iter):
            imgs, masks = imgs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if tqdm:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")
            elif batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0.0

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", unit="batch") if tqdm else val_loader
        with torch.no_grad():
            for imgs, masks in val_iter:
                imgs, masks = imgs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.long)
                val_loss += criterion(model(imgs), masks).item()

                if tqdm:
                    val_iter.set_postfix(val_loss=f"{val_loss / (val_iter.n if hasattr(val_iter, 'n') else 1):.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    print("Starting final evaluation on validation set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device, dtype=torch.float32)
            preds = torch.max(model(imgs), 1)[1]
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(masks.numpy().flatten())

    a_p = np.concatenate(all_preds)
    a_l = np.concatenate(all_labels)

    acc = accuracy_score(a_l, a_p)
    iou = jaccard_score(a_l, a_p, average='macro')
    f1 = f1_score(a_l, a_p, average='macro')

    return acc, iou, f1