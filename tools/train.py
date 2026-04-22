import os
import sys
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from core.dataset import TunnelDataset, get_transforms
from core.model import MDCNet
from core.loss import RecallFocusedComboLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def calculate_metrics_tensor(preds, masks):
    preds = (torch.sigmoid(preds) > 0.5).long()
    masks = masks.long()

    tp = (preds & masks).sum().float()
    fp = (preds & ~masks).sum().float()
    fn = (~preds & masks).sum().float()
    return tp, fp, fn


def calculate_f_beta(precision, recall, beta=1.5):
    beta_sq = beta ** 2
    if (precision + recall) == 0: return 0.0
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"启动训练引擎，当前设备: {device} | 模式: {args.model}")

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_dir, "train", "images")
    train_mask_dir = os.path.join(args.data_dir, "train", "masks")
    val_dir = os.path.join(args.data_dir, "val", "images")
    val_mask_dir = os.path.join(args.data_dir, "val", "masks")

    train_ds = TunnelDataset(train_dir, train_mask_dir, transform=get_transforms('train'))
    val_ds = TunnelDataset(val_dir, val_mask_dir, transform=get_transforms('valid'))

    if len(train_ds) == 0:
        logger.error(f"找不到训练数据，请检查路径: {train_dir}")
        return
    if len(val_ds) == 0:
        logger.error(f"找不到验证数据，请检查路径: {val_dir}")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MDCNet(mode=args.model).to(device)

    scaler = torch.amp.GradScaler('cuda')

    criterion = RecallFocusedComboLoss(pos_weight_val=5.0, tversky_alpha=0.3, tversky_beta=0.7).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    accumulation_steps = max(1, args.target_batch // args.batch_size)
    best_f2 = 0.0

    logger.info(f"物理 Batch Size: {args.batch_size}, 逻辑(累加) Batch Size: {args.target_batch}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs}", leave=False)

        for i, (images, masks, _) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast('cuda'):
                preds = model(images)
                loss = criterion(preds, masks)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # 尾部对齐保护
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            loop.set_postfix(loss=loss.item() * accumulation_steps)

        # Validation Loop
        model.eval()
        total_tp = torch.tensor(0.0, device=device)
        total_fp = torch.tensor(0.0, device=device)
        total_fn = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.autocast('cuda'):
                    preds = model(images)
                tp, fp, fn = calculate_metrics_tensor(preds, masks)
                total_tp += tp;
                total_fp += fp;
                total_fn += fn

        total_tp, total_fp, total_fn = total_tp.item(), total_fp.item(), total_fn.item()
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)

        f2_score = calculate_f_beta(precision, recall, beta=1.5)
        logger.info(f"Ep {epoch + 1:03d} | F2: {f2_score:.4f} | IoU: {iou:.4f} | Rec: {recall:.4f}")

        if f2_score > best_f2:
            best_f2 = f2_score
            save_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"新的最高 F2 分数产生！权重已保存至: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours', help='ours, ablation_no_dlka, ablation_no_coord')
    parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'dummy_data'))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--target_batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default=os.path.join(ROOT_DIR, 'weights'))
    args = parser.parse_args()

    train(args)