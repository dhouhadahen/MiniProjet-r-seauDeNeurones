"""Entraîne le détecteur, évalue (IoU moyen, % IoU > 0.5) et sauvegarde une figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.detector import Detector, ShapeDataset, compute_loss, iou, predict


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train() # Permet de passer le modèle en mode entraînement (active dropout, batchnorm, etc.)
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()  # Réinitialise les gradients
        outputs = model(images)  # Passe les images dans le modèle pour obtenir les prédictions
        loss = compute_loss(outputs, targets)  # Calcule la perte entre les prédictions et les cibles
        loss.backward()  # Calcule les gradients
        optimizer.step()  # Met à jour les poids du modèle
        total_loss += loss.item()  # Accumule la perte pour calculer la moyenne plus tard
    # A COMPLETER Aide : Bouclez sur les images et cibles du loader, calculez la loss, 
    # faites un backward et un step de l'optimizer.
    return total_loss / len(loader.dataset)


def evaluate(model: Detector, loader: DataLoader, size: int) -> tuple[float, float]:
    from src.detector import K, ANCHORS, STRIDE, iou, predict
    
    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            
            for logit, target in zip(logits, targets):
                #predict renvoie maintenant une liste de boîtes.
                # On prend la première (la plus confiante) pour le calcul du score d'évaluation.
                pred_boxes, pred_scores = predict(logit, size)
                if len(pred_boxes) > 0:
                    box = pred_boxes[0] # On extrait la meilleure boîte
                else:
                    box = torch.zeros(4, device=logits.device)
                
                # 2. Reconstruction de TOUTES les vraies boîtes depuis les 15 canaux
                target = target.view(K, 5, grid, grid)
                true_boxes = []
                
                for k in range(K):
                    for row in range(grid):
                        for col in range(grid):
                            if target[k, 0, row, col] > 0.5: # Si un objet est présent
                                dx = target[k, 1, row, col]
                                dy = target[k, 2, row, col]
                                tw = target[k, 3, row, col]
                                th = target[k, 4, row, col]
                                
                                cx = (col + dx) * STRIDE
                                cy = (row + dy) * STRIDE
                                w = tw * ANCHORS[k, 0] * size
                                h = th * ANCHORS[k, 1] * size
                                
                                true_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                
                # 3. Calcul de l'IoU avec la meilleure boîte
                if true_boxes and len(pred_boxes) > 0:
                    truth_tensor = torch.tensor(true_boxes, device=box.device)
                    iou_scores = iou(box, truth_tensor)
                    ious.append(float(iou_scores.max()))
                else:
                    ious.append(0.0)
                    
    t = torch.tensor(ious)
    return t.mean().item(), (t > 0.5).float().mean().item()