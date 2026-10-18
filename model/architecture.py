"""
model/architecture.py
=====================
DeepFake Detection Model — Hybrid CNN-LSTM with CBAM + Frequency Head
----------------------------------------------------------------------
Streams:
  1. EfficientNet-B4 backbone  → spatial/semantic features (1792-dim)
  2. CBAM attention            → channel + spatial artifact localization
  3. Frequency analysis head   → Laplacian high-freq GAN fingerprint detection
  4. LSTM layer                → temporal modeling across video frame sequences
  5. Fusion classifier         → Dropout → Linear(512) → GELU → Linear(128) → Softmax(2)

Output: logits [real_score, fake_score]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights
import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────
#  1. Squeeze-and-Excitation Block (Channel Attention)
# ─────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """
    Channel-wise attention: learns which feature channels are most
    informative for detecting deepfake artifacts.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)        # Global average pool → (B, C)
        w = self.fc(w).view(b, c, 1, 1)    # Excitation → (B, C, 1, 1)
        return x * w                        # Channel-wise scaling


# ─────────────────────────────────────────────────────────────────
#  2. Spatial Attention Module
# ─────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial attention: highlights spatial regions with manipulation
    artifacts (e.g., face boundary, eye, mouth regions in deepfakes).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool across channel dim to get spatial context
        avg = torch.mean(x, dim=1, keepdim=True)           # (B, 1, H, W)
        mx, _ = torch.max(x, dim=1, keepdim=True)          # (B, 1, H, W)
        concat = torch.cat([avg, mx], dim=1)                # (B, 2, H, W)
        attn = self.sigmoid(self.conv(concat))              # (B, 1, H, W)
        return x * attn


# ─────────────────────────────────────────────────────────────────
#  3. CBAM: Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    """
    Combines channel attention (SEBlock) and spatial attention
    sequentially for comprehensive artifact localization.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = SEBlock(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)   # Step 1: focus on relevant channels
        x = self.spatial_attn(x)   # Step 2: focus on relevant spatial regions
        return x


# ─────────────────────────────────────────────────────────────────
#  4. Frequency Analysis Head
# ─────────────────────────────────────────────────────────────────
class FrequencyAnalysisHead(nn.Module):
    """
    Detects high-frequency GAN fingerprints invisible to the human eye.

    Deepfake generators (StyleGAN, FaceSwap, etc.) leave characteristic
    artifacts in the high-frequency domain. We use a Laplacian edge
    filter to extract these frequencies, then classify them.

    Output: 128-dim feature vector
    """
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 380 → 190

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 190 → 95

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                        # → 4×4
        )
        self.fc = nn.Linear(128 * 16, out_features)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Laplacian filter to extract high-frequency components.
        Converts RGB to grayscale first, then applies the filter.
        """
        # RGB → Grayscale (ITU-R BT.601 weights)
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # Laplacian kernel for edge/high-frequency detection
        kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)

        freq = F.conv2d(gray, kernel, padding=1)
        return freq  # (B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self._laplacian(x)               # High-freq map
        feat = self.conv_layers(freq)           # CNN on freq map
        feat = feat.view(feat.size(0), -1)      # Flatten
        return self.fc(feat)                    # (B, 128)


# ─────────────────────────────────────────────────────────────────
#  5. Main DeepFake Detector
# ─────────────────────────────────────────────────────────────────
class DeepFakeDetector(nn.Module):
    """
    Full hybrid CNN-LSTM deepfake detection model.

    For images:  single frame forward pass
    For videos:  sequence of frames through LSTM for temporal modeling

    Architecture:
        Frame(s) → EfficientNet-B4 → CBAM → GlobalPool → (B, 1792)
                 → Frequency Head             → (B, 128)
                 → Concat                     → (B, 1920)
                 → LSTM (video) / Identity (image)
                 → Classifier: 1920→512→128→2
                 → Softmax → [P(real), P(fake)]
    """

    BACKBONE_OUT = 1792   # EfficientNet-B4 output channels

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.4,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
    ):
        super().__init__()

        # ── EfficientNet-B4 Backbone ──────────────────────────────
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)
        self.features = backbone.features   # Output: (B, 1792, H, W)

        # ── CBAM Attention ────────────────────────────────────────
        self.cbam = CBAM(self.BACKBONE_OUT)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Frequency Analysis Head ───────────────────────────────
        self.freq_head = FrequencyAnalysisHead(out_features=128)

        # Fusion feature size: CNN(1792) + Freq(128)
        self.fusion_dim = self.BACKBONE_OUT + 128

        # ── LSTM for Temporal Modeling (video frames) ─────────────
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # ── Fusion Classifier ─────────────────────────────────────
        # Handles both image (fusion_dim) and video (lstm_hidden) input
        self.classifier_image = self._build_classifier(self.fusion_dim, dropout)
        self.classifier_video = self._build_classifier(lstm_hidden, dropout)

        # ── Grad-CAM hooks ────────────────────────────────────────
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.features[-1].register_forward_hook(self._save_activations)
        self.features[-1].register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _build_classifier(in_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),   # [real_logit, fake_logit]
        )

    # ── Grad-CAM Hook Callbacks ───────────────────────────────────
    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── Single Frame Feature Extraction ──────────────────────────
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract fused feature vector from a single frame.
        x: (B, 3, H, W)
        Returns: (B, fusion_dim)
        """
        # Stream 1+2: Backbone + CBAM
        feat_map = self.features(x)             # (B, 1792, H, W)
        feat_map = self.cbam(feat_map)
        spatial = self.global_pool(feat_map).squeeze(-1).squeeze(-1)  # (B, 1792)

        # Stream 3: Frequency analysis
        freq = self.freq_head(x)                # (B, 128)

        # Fuse both streams
        return torch.cat([spatial, freq], dim=1)  # (B, 1920)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "image",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:    Image: (B, 3, H, W)
                  Video: (B, T, 3, H, W)  where T = num frames
            mode: 'image' or 'video'

        Returns:
            logits: (B, 2)
        """
        if mode == "image":
            fused = self._extract_features(x)           # (B, 1920)
            return self.classifier_image(fused)

        elif mode == "video":
            B, T, C, H, W = x.shape
            # Process all frames at once by merging B and T dims
            x_flat = x.view(B * T, C, H, W)
            fused_flat = self._extract_features(x_flat) # (B*T, 1920)
            fused_seq = fused_flat.view(B, T, -1)       # (B, T, 1920)

            # LSTM over frame sequence
            lstm_out, _ = self.lstm(fused_seq)          # (B, T, 512)
            # Use last timestep output as video representation
            video_feat = lstm_out[:, -1, :]             # (B, 512)
            return self.classifier_video(video_feat)

        else:
            raise ValueError(f"mode must be 'image' or 'video', got '{mode}'")

    # ── Inference Helpers ─────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mode: str = "image",
    ) -> Tuple[str, float, float]:
        """
        Returns: (label, fake_prob, real_prob)
        """
        self.eval()
        logits = self(x, mode=mode)
        probs = F.softmax(logits, dim=1)
        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()
        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        return label, fake_prob, real_prob

    def grad_cam(
        self,
        x: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap (requires single image input).

        Args:
            x:            (1, 3, H, W) — single image tensor with grad enabled
            target_class: 0=real, 1=fake

        Returns:
            heatmap: np.ndarray (H, W) — values in [0, 1]
        """
        self.eval()
        x = x.clone().requires_grad_(True)

        logits = self(x, mode="image")
        self.zero_grad()
        logits[0, target_class].backward()

        grads = self._gradients    # (1, C, h, w)
        acts  = self._activations  # (1, C, h, w)

        if grads is None or acts is None:
            return np.zeros((x.shape[2], x.shape[3]))

        # Weight each activation map by its gradient importance
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True) # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ─────────────────────────────────────────────────────────────────
#  Factory function
# ─────────────────────────────────────────────────────────────────
def build_model(
    pretrained: bool = True,
    dropout: float = 0.4,
    lstm_hidden: int = 512,
    lstm_layers: int = 2,
) -> DeepFakeDetector:
    """Build and return DeepFakeDetector with given config."""
    return DeepFakeDetector(
        pretrained=pretrained,
        dropout=dropout,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    )
