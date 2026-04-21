import os
import argparse
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import WeatherDataset
from model import WeatherCNN

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def focal_loss_with_logits(preds, targets, alpha=0.98, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    pt = torch.exp(-bce_loss)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def zero_loss_like(preds):
    return preds.sum() * 0.0


def custom_loss(preds, targets, target_mean=None, target_std=None):
    valid_base = ~torch.isnan(targets[:, :5])
    if valid_base.any():
        mse_base = F.mse_loss(preds[:, :5][valid_base], targets[:, :5][valid_base])
    else:
        mse_base = zero_loss_like(preds)

    mask_apcp = ~torch.isnan(targets[:, 5])
    if mask_apcp.any():
        mse_apcp = F.smooth_l1_loss(preds[mask_apcp, 5], targets[mask_apcp, 5])
    else:
        mse_apcp = zero_loss_like(preds)

    valid_bin = ~torch.isnan(targets[:, 6])
    if valid_bin.any():
        bce = focal_loss_with_logits(preds[valid_bin, 6], targets[valid_bin, 6])
    else:
        bce = zero_loss_like(preds)

    total = mse_base + (0.15 * mse_apcp) + (100.0 * bce)
    return total, mse_base, mse_apcp, bce


def validate(model, val_loader, device, target_mean=None, target_std=None):
    model.eval()

    total_loss = 0.0
    total_mse_base = 0.0
    total_mse_apcp = 0.0
    valid_samples = 0
    valid_apcp_samples = 0
    all_preds_bin = []
    all_targets_bin = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            valid_mask = ~torch.isnan(x).view(x.size(0), -1).any(dim=1)
            x = x[valid_mask]
            y = y[valid_mask]

            if x.size(0) == 0:
                continue

            preds = model(x)
            loss, mse_base, mse_apcp, _ = custom_loss(preds, y, target_mean, target_std)

            bs = x.size(0)
            total_loss += loss.item() * bs
            valid_samples += bs

            if target_mean is not None and target_std is not None:
                reg_mean = target_mean[:6]
                reg_std = target_std[:6]
                unnorm_preds = preds[:, :6] * reg_std + reg_mean
                unnorm_y = y[:, :6] * reg_std + reg_mean

                valid_base = ~torch.isnan(unnorm_y[:, :5])
                if valid_base.any():
                    true_mse_base = F.mse_loss(
                        unnorm_preds[:, :5][valid_base], unnorm_y[:, :5][valid_base]
                    )
                    total_mse_base += true_mse_base.item() * bs

                mask_apcp = (unnorm_y[:, 5] > 2.0) & (~torch.isnan(unnorm_y[:, 5]))
                apcp_count = mask_apcp.sum().item()
                if apcp_count > 0:
                    true_mse_apcp = F.mse_loss(unnorm_preds[mask_apcp, 5], unnorm_y[mask_apcp, 5])
                    total_mse_apcp += true_mse_apcp.item() * apcp_count
                    valid_apcp_samples += apcp_count
            else:
                total_mse_base += mse_base.item() * bs
                mask_apcp = (y[:, 5] > 2.0) & (~torch.isnan(y[:, 5]))
                apcp_count = mask_apcp.sum().item()
                if apcp_count > 0:
                    total_mse_apcp += mse_apcp.item() * apcp_count
                    valid_apcp_samples += apcp_count

            valid_bin = ~torch.isnan(y[:, 6])
            if valid_bin.any():
                all_preds_bin.extend(torch.sigmoid(preds[valid_bin, 6]).cpu().tolist())
                all_targets_bin.extend(y[valid_bin, 6].cpu().tolist())

    avg_loss = total_loss / valid_samples if valid_samples > 0 else 0.0
    rmse_base = (total_mse_base / valid_samples) ** 0.5 if valid_samples > 0 else 0.0
    rmse_apcp = (total_mse_apcp / valid_apcp_samples) ** 0.5 if valid_apcp_samples > 0 else 0.0

    try:
        auc = roc_auc_score(all_targets_bin, all_preds_bin)
    except ValueError:
        auc = 0.5

    return avg_loss, rmse_base, rmse_apcp, auc


def main():
    parser = argparse.ArgumentParser(description="Train WeatherCNN")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--metrics_file", type=str, default="metrics.json", help="File to save metrics")
    parser.add_argument("--save_dir", type=str, default="/cluster/tufts/c26sp1cs0137/cgolem01", help="Directory to save model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = WeatherDataset(years_to_include=["2018", "2019"])
    val_dataset = WeatherDataset(years_to_include=["2020"])

    target_mean = getattr(train_dataset, "target_mean", None)
    target_std = getattr(train_dataset, "target_std", None)

    if target_mean is not None and target_std is not None:
        target_mean = target_mean.to(device)
        target_std = target_std.to(device)
        val_dataset.target_mean = train_dataset.target_mean
        val_dataset.target_std = train_dataset.target_std

    sample_x, _ = train_dataset[0]
    in_channels = sample_x.shape[0]
    print(f"Detected input channels: {in_channels}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = WeatherCNN(in_channels=in_channels, out_channels=7, use_residual=args.use_residual).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    epochs = args.epochs
    best_val_auc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 8
    save_dir = args.save_dir

    all_metrics = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse_base_total = 0.0
        train_mse_apcp_total = 0.0
        train_bce_total = 0.0
        valid_train_samples = 0
        valid_apcp_samples = 0

        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)

            valid_mask = ~torch.isnan(x).view(x.size(0), -1).any(dim=1)
            x = x[valid_mask]
            y = y[valid_mask]

            if x.size(0) < 2:
                continue

            optimizer.zero_grad()
            preds = model(x)
            loss, mse_base, mse_apcp, bce = custom_loss(preds, y, target_mean, target_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_mse_base_total += mse_base.item() * bs
            train_bce_total += bce.item() * bs
            valid_train_samples += bs

            if target_mean is not None and target_std is not None:
                apcp_unnorm = y[:, 5] * target_std[5] + target_mean[5]
            else:
                apcp_unnorm = y[:, 5]

            mask_apcp = (apcp_unnorm > 2.0) & (~torch.isnan(y[:, 5]))
            apcp_count = mask_apcp.sum().item()
            if apcp_count > 0:
                train_mse_apcp_total += mse_apcp.item() * apcp_count
                valid_apcp_samples += apcp_count

            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / valid_train_samples if valid_train_samples > 0 else 0.0
        avg_base = train_mse_base_total / valid_train_samples if valid_train_samples > 0 else 0.0
        avg_apcp = train_mse_apcp_total / valid_apcp_samples if valid_apcp_samples > 0 else 0.0
        avg_bce = train_bce_total / valid_train_samples if valid_train_samples > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train Total Loss: {avg_train_loss:.4f} | "
            f"Base MSE: {avg_base:.4f} | APCP MSE: {avg_apcp:.4f} | BCE: {avg_bce:.4f}"
        )

        val_loss, val_rmse_base, val_rmse_apcp, val_auc = validate(
            model, val_loader, device, target_mean, target_std
        )
        print(f"Epoch [{epoch + 1}/{epochs}] - Val Total Loss: {val_loss:.4f}")
        print(
            f"Metrics -> True scale Base RMSE: {val_rmse_base:.4f} | "
            f"True scale APCP RMSE: {val_rmse_apcp:.4f} | Val AUC: {val_auc:.4f}"
        )

        all_metrics.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_rmse_base": val_rmse_base,
            "val_rmse_apcp": val_rmse_apcp,
            "val_auc": val_auc
        })

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            # If using custom save_dir/metrics_file, prefix model name might be useful, 
            # but we keep simple here.
            save_path = os.path.join(save_dir, f"best_model_{val_auc:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"Early stopping patience: {epochs_no_improve}/{early_stop_patience}")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered. Training stopped.")
                break

    with open(args.metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved metrics to {args.metrics_file}")


if __name__ == "__main__":
    main()
