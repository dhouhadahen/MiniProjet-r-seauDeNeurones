"""Détecteur minimal : feature maps -> prédiction par cellule -> NMS -> une boîte.

Image 64x64 -> 3 blocs conv stride 2 -> feature map 32x8x8 (stride 8).
Une conv 1x1 prédit pour chaque cellule 5 valeurs : (objectness, dx, dy, w, h)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : taille relative à l'image
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8

K = 3
ANCHORS = torch.tensor([
    [0.25, 0.25],  # Ancre 0 : Carré moyen
    [0.40, 0.15],  # Ancre 1 : Rectangle horizontal
    [0.15, 0.40]   # Ancre 2 : Rectangle vertical
])

def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4)."""
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 1.1) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence et retourne les indices des boîtes conservées."""
    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        best = order[0]
        kept.append(int(best))
        order = order[1:]
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept


def code(cx: float, cy: float, w: float, h: float, size: int = 64) -> torch.Tensor:
    grid = size // STRIDE
    gx, gy = cx / size * grid, cy / size * grid
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
    
    # Cible : K * 5 valeurs (15) au lieu de 5
    target = torch.zeros(K * 5, grid, grid)
    
    w_norm, h_norm = w / size, h / size
    
    best_iou = 0
    best_anchor_idx = 0
    
    # Choix de la meilleure ancre basée sur la forme (largeur/hauteur)
    for i in range(K):
        anchor_w, anchor_h = ANCHORS[i]
        inter_w = min(w_norm, anchor_w.item())
        inter_h = min(h_norm, anchor_h.item())
        inter_area = inter_w * inter_h
        union_area = (w_norm * h_norm) + (anchor_w.item() * anchor_h.item()) - inter_area
        iou_score = inter_area / union_area
        
        if iou_score > best_iou:
            best_iou = iou_score
            best_anchor_idx = i
            
    start_idx = best_anchor_idx * 5
    
    target[start_idx : start_idx + 5, row, col] = torch.tensor([
        1.0, 
        gx - col, 
        gy - row, 
        w_norm / ANCHORS[best_anchor_idx, 0], 
        h_norm / ANCHORS[best_anchor_idx, 1]
    ])
    
    return target

def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """(15, G, G) -> boîtes (K*G*G, 4) en pixels [x1, y1, x2, y2] et scores (K*G*G,)."""
    grid = logits.shape[-1]
    
    # 1. On réorganise les 15 canaux en 3 tiroirs (ancres) de 5 valeurs
    # Le tenseur passe de [15, 8, 8] à [3, 5, 8, 8]
    logits = logits.view(K, 5, grid, grid)
    p = torch.sigmoid(logits)
    
    # 2. Grille de coordonnées de base
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    
    # 3. Calcul des centres (identique, mais appliqué sur les 3 ancres à la fois)
    cx = (cols + p[:, 1]) * STRIDE
    cy = (rows + p[:, 2]) * STRIDE
    
    # 4. L'astuce du décodage : multiplier par la taille du gabarit
    # On aligne les tailles des ancres (K, 2) pour le calcul mathématique (K, 1, 1)
    anchor_w = ANCHORS[:, 0].view(K, 1, 1)
    anchor_h = ANCHORS[:, 1].view(K, 1, 1)
    
    # On retrouve la vraie largeur en pixels : ratio_prédit * taille_ancre * 64
    w = p[:, 3] * anchor_w * size
    h = p[:, 4] * anchor_h * size
    
    # 5. Construction des coins des boîtes
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    
    # 6. On aplatit tout. Au lieu d'avoir 64 boîtes, on a désormais 3 * 64 = 192 boîtes possibles.
    return boxes.reshape(-1, 4), p[:, 0].reshape(-1)

class ShapeDataset(Dataset):
    """Cible (5, G, G) : tout à 0 sauf la cellule contenant le centre."""

    def __init__(self, folder: Path, size: int = 64) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator).")
        self.grid = size // STRIDE

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        cx, cy, w, h = map(float, path.with_suffix(".txt").read_text().split())
        target = code(cx, cy, w, h, size=image.shape[-1])
        return image, target


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        blocks = [nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(32, K*5, kernel_size=1, stride = 1)
        # A_COMPLETER Aide : Définir les couches du détecteur (backbone et head).
        

    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.backbone(images)
        # A_COMPLETER Aide : Appliquer le backbone sur les images pour obtenir les feature maps.
        
        return feature_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.feature_maps(images)
        output = self.head(feature_maps)
        # A_COMPLETER Aide : Appliquer la tête sur les feature maps pour obtenir les logits de sortie.
        
        return output





def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE sur l'objectness de toutes les ancres + MSE sur la boîte positive."""
    
    # B = Batch (64), C = Canaux (15), H = Hauteur (8), W = Largeur (8)
    B, C, H, W = logits.shape
    
    # On redécoupe les 15 canaux en 3 ancres de 5 canaux. 
    # Le tenseur passe de [B, 15, 8, 8] à [B, 3, 5, 8, 8]
    logits = logits.view(B, K, 5, H, W)
    target = target.view(B, K, 5, H, W)
    
    # 1. Calcul de la Loss Ojectness (BCE) sur le canal 0 de TOUTES les ancres
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, :, 0], target[:, :, 0], pos_weight=torch.tensor(10.0)
    )
    
    # 2. On crée un masque pour trouver la seule ancre (parmi les 3) qui contient la vraie figure
    positive = target[:, :, 0] > 0.5 
    
    # 3. On extrait les prédictions et les cibles des coordonnées (canaux 1 à 4) UNIQUEMENT pour l'ancre positive
    pred_box = torch.sigmoid(logits[:, :, 1:]).permute(0, 1, 3, 4, 2)[positive]
    true_box = target[:, :, 1:].permute(0, 1, 3, 4, 2)[positive]
    
    # 4. Calcul de la Loss Coordonnées (MSE) si au moins un objet est présent
    if positive.any():
        box_loss = nn.functional.mse_loss(pred_box, true_box)
    else:
        box_loss = torch.tensor(0.0, device=logits.device)
        
    return obj_loss + 10.0 * box_loss




def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Plusieurs figures par image : on garde toutes les boîtes avec un score > 50% après NMS."""
    boxes, scores = decode(logits, size)
    
    # 1. On élimine toutes les boîtes où le réseau n'est pas confiant (score < 50%)
    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    
    # 2. Sécurité : si aucune boîte n'est trouvée, on renvoie des tenseurs vides
    if len(boxes) == 0:
        return torch.zeros((0, 4), device=logits.device), torch.zeros(0, device=logits.device)
        
    # 3. NMS (Non-Maximum Suppression) : On supprime les boîtes en double 
    # qui encadreraient le même cercle (seuil de superposition à 30%)
    keep_indices = nms(boxes, scores, iou_threshold=0.3)
    
    return boxes[keep_indices], scores[keep_indices]
