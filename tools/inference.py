import os
import sys
import torch
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from core.model import MDCNet


class OptimizedInferenceEngine:

    def __init__(self, crack_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"启动工业级推理引擎，加载设备: {self.device}")

        self.crack_model = MDCNet(num_classes=1, mode='ours').to(self.device)

        try:
            state_dict = torch.load(crack_model_path, map_location=self.device)
            self.crack_model.load_state_dict(state_dict, strict=False)
            self.crack_model.eval()
            print(f"裂隙模型权重加载成功: {os.path.basename(crack_model_path)}")
        except Exception as e:
            print(f"警告：权重加载失败！{e}\n请确保你已经训练并生成了 best_ours.pth")

    def _get_gaussian_weight_map(self, size):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        d = np.sqrt(xx * xx + yy * yy)
        sigma, mu = 0.5, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        g = (g - g.min()) / (g.max() - g.min())
        return torch.from_numpy(g).float().to(self.device)

    @torch.no_grad()
    def _sliding_window_inference(self, img, smooth_sigma=0):
        patch_size = 512
        stride = 256
        h, w = img.shape[:2]

        pad_h = max(0, patch_size - h) if h < patch_size else (patch_size - h % stride) % stride
        pad_w = max(0, patch_size - w) if w < patch_size else (patch_size - w % stride) % stride
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        h_pad, w_pad = img_padded.shape[:2]

        prob_map_gpu = torch.zeros((h_pad, w_pad), dtype=torch.float32, device=self.device)
        weight_map_sum_gpu = torch.zeros((h_pad, w_pad), dtype=torch.float32, device=self.device)
        gaussian_weight = self._get_gaussian_weight_map(patch_size)

        patches_coords = [(y, x) for y in range(0, h_pad - patch_size + 1, stride)
                          for x in range(0, w_pad - patch_size + 1, stride)]

        batch_size = 2

        img_padded_tensor = torch.from_numpy(img_padded).to(self.device).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img_padded_tensor = (img_padded_tensor - mean) / std

        for i in range(0, len(patches_coords), batch_size):
            batch_data = patches_coords[i: i + batch_size]
            batch_tensor = torch.stack([
                img_padded_tensor[:, y:y + patch_size, x:x + patch_size] for y, x in batch_data
            ])

            with torch.autocast('cuda'):
                logits = self.crack_model(batch_tensor)
                preds = torch.sigmoid(logits).squeeze(1)

            for j, (y, x) in enumerate(batch_data):
                weighted_pred = preds[j] * gaussian_weight
                prob_map_gpu[y:y + patch_size, x:x + patch_size] += weighted_pred
                weight_map_sum_gpu[y:y + patch_size, x:x + patch_size] += gaussian_weight

        final_prob_gpu = prob_map_gpu / torch.clamp(weight_map_sum_gpu, min=1e-6)
        final_prob = final_prob_gpu[:h, :w].cpu().numpy()

        if smooth_sigma > 0:
            k = int(smooth_sigma * 4) + 1
            if k % 2 == 0: k += 1
            final_prob = cv2.GaussianBlur(final_prob, (k, k), smooth_sigma)

        return final_prob

    def predict_full_pipeline(self, img_bgr, use_ellipse_fit=True):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crack_prob = self._sliding_window_inference(img_rgb)
        crack_mask = (crack_prob > 0.5).astype(np.uint8) * 255

        h, w = img_bgr.shape[:2]
        face_mask = np.ones((h, w), dtype=np.uint8) * 255

        return crack_mask, face_mask