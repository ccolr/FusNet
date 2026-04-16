import argparse
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


MEAN = [55.7578, 67.4502, 58.6568]
STD = [37.5201, 34.2345, 30.3007]
CLASS_NAMES = ["Non-Bamboo", "Bamboo"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(224, 224),
                scale=(0.65, 1.0),
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=45,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.15),
                    A.GridDistortion(num_steps=5, distort_limit=0.2),
                    A.ElasticTransform(alpha=1.0, sigma=40.0),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=15),
                    A.CLAHE(clip_limit=4.0),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.07)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.Sharpen(alpha=(0.1, 0.3)),
                ],
                p=0.4,
            ),
            A.CoarseDropout(
                num_holes_range=(4, 14),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                fill_mask=0,
                p=0.35,
            ),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


class BambooDataset(Dataset):
    def __init__(self, data_dir: str, list_path: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with open(list_path, "r", encoding="utf-8") as f:
            img_ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for rel_img_path in img_ids:
            img_path = os.path.join(data_dir, rel_img_path)
            mask_path = os.path.join(data_dir, rel_img_path.replace("images", "labels").replace(".tif", "_mask.tif"))
            self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        with rasterio.open(img_path) as src:
            image = src.read()  # (C, H, W)
        image = np.moveaxis(image, 0, -1).astype(np.uint8)  # (H, W, C)
        if image.shape[2] > 3:
            image = image[:, :, :3]

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


def get_binary_logits(logits: torch.Tensor):
    if logits.ndim != 4:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    if logits.shape[1] == 1:
        return logits[:, 0, :, :]
    if logits.shape[1] == 2:
        return logits[:, 1, :, :]
    raise ValueError(f"Expected channel size 1 or 2, but got {logits.shape[1]}")


def confusion_stats(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, tn, fp, fn


def metrics_from_confusion(tp: float, tn: float, fp: float, fn: float):
    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    oa = accuracy
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou_pos = tp / (tp + fp + fn + eps)
    iou_neg = tn / (tn + fp + fn + eps)
    miou = (iou_pos + iou_neg) / 2.0
    return {
        "acc": accuracy,
        "oa": oa,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "miou": miou,
    }


def plot_curve(train_values, val_values, ylabel, title, save_path):
    epochs = np.arange(1, len(train_values) + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.plot(epochs, train_values, color="#1f77b4", linewidth=2.2, label="Train")
    ax.plot(epochs, val_values, color="#d62728", linewidth=2.2, label="Validation")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, save_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    thresh = cm.max() * 0.5 if cm.max() > 0 else 0.5
    cm_sum = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm = cm / cm_sum
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    bce_loss,
    dice_loss,
    device,
    train_mode: bool,
    amp_enabled: bool,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
    n_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        n_samples += images.size(0)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            binary_logits = get_binary_logits(logits)

            bce = bce_loss(binary_logits, masks)
            dice = dice_loss(binary_logits, masks)
            loss = 0.8 * bce + 0.2 * dice

        if train_mode:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        probs = torch.sigmoid(binary_logits)
        preds = (probs >= 0.5).long()
        gt = (masks >= 0.5).long()

        tp, tn, fp, fn = confusion_stats(preds, gt)
        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / max(n_samples, 1)
    metrics = metrics_from_confusion(tp_total, tn_total, fp_total, fn_total)
    return avg_loss, metrics


def evaluate_confusion_matrix(model, loader, device):
    model.eval()
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            binary_logits = get_binary_logits(logits)
            preds = (torch.sigmoid(binary_logits) >= 0.5).long()
            gt = (masks >= 0.5).long()
            tp, tn, fp, fn = confusion_stats(preds, gt)
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn
    return np.array([[tn_total, fp_total], [fn_total, tp_total]], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Train FusNet for bamboo binary segmentation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=".", help="Data root directory")
    args = parser.parse_args()

    set_seed(42)

    try:
        from model.FusNet import FusNet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Import FusNet failed. Please ensure required dependencies are installed (for example `mamba_ssm`)."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    num_workers = min(8, os.cpu_count() if os.cpu_count() is not None else 2)

    output_dir = os.path.join("fusnet_outputs")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    best_weight_path = os.path.join(output_dir, "best_model.pth")
    last_weight_path = os.path.join(output_dir, "last_model.pth")
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    acc_curve_path = os.path.join(output_dir, "accuracy_curve.png")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    train_dataset = BambooDataset(
        data_dir=args.data_dir,
        list_path=os.path.join(args.data_dir, "train.txt"),
        transform=get_train_transforms(),
    )
    valid_dataset = BambooDataset(
        data_dir=args.data_dir,
        list_path=os.path.join(args.data_dir, "valid.txt"),
        transform=get_valid_transforms(),
    )

    total = len(train_dataset) + len(valid_dataset)
    split_ratio = len(train_dataset) / max(total, 1)
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Train ratio: {split_ratio:.3f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = FusNet(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=max(args.lr * 0.01, 1e-6),
    )
    scaler = GradScaler(enabled=amp_enabled)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_f1 = -1.0
    best_epoch = -1

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "Epoch,LR,"
            "Train_Loss,Val_Loss,Train_Acc,Val_Acc,"
            "Train_mIoU,Val_mIoU,Train_F1,Val_F1,Train_Recall,Val_Recall,Train_OA,Val_OA,Train_Dice,Val_Dice\n"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            device=device,
            train_mode=True,
            amp_enabled=amp_enabled,
        )

        val_loss, val_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=optimizer,
            scaler=scaler,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            device=device,
            train_mode=False,
            amp_enabled=amp_enabled,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.8f},"
                f"{train_loss:.6f},{val_loss:.6f},{train_metrics['acc']:.6f},{val_metrics['acc']:.6f},"
                f"{train_metrics['miou']:.6f},{val_metrics['miou']:.6f},{train_metrics['f1']:.6f},{val_metrics['f1']:.6f},"
                f"{train_metrics['recall']:.6f},{val_metrics['recall']:.6f},{train_metrics['oa']:.6f},{val_metrics['oa']:.6f},"
                f"{train_metrics['dice']:.6f},{val_metrics['dice']:.6f}\n"
            )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"LR={current_lr:.6e} | "
            f"Train: Loss={train_loss:.4f}, Acc={train_metrics['acc']:.4f}, mIoU={train_metrics['miou']:.4f}, "
            f"F1={train_metrics['f1']:.4f}, Recall={train_metrics['recall']:.4f}, OA={train_metrics['oa']:.4f}, Dice={train_metrics['dice']:.4f} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_metrics['acc']:.4f}, mIoU={val_metrics['miou']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, OA={val_metrics['oa']:.4f}, Dice={val_metrics['dice']:.4f}"
        )

        plot_curve(
            train_values=history["train_loss"],
            val_values=history["val_loss"],
            ylabel="Loss",
            title="Training and Validation Loss",
            save_path=loss_curve_path,
        )
        plot_curve(
            train_values=history["train_acc"],
            val_values=history["val_acc"],
            ylabel="Accuracy",
            title="Training and Validation Accuracy",
            save_path=acc_curve_path,
        )

        torch.save(model.state_dict(), last_weight_path)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_weight_path)

    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")

    if os.path.exists(best_weight_path):
        state_dict = torch.load(best_weight_path, map_location=device)
        model.load_state_dict(state_dict)

    cm = evaluate_confusion_matrix(model, valid_loader, device)
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    print("Training complete.")
    print(f"Outputs saved to: {output_dir}")
    print(f"Log file: {log_path}")
    print(f"Best weight: {best_weight_path}")
    print(f"Last weight: {last_weight_path}")
    print(f"Loss curve: {loss_curve_path}")
    print(f"Accuracy curve: {acc_curve_path}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
