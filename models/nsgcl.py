"""Integrated Neuro-Symbolic Graph Contrastive Learning model."""
import torch
from torch import nn
import torch.nn.functional as F
from .spatiotemporal_encoder import NSGCLRepresentationEncoder
from .contrastive import ContrastiveProjectionHead, InfoNCELoss


class NSGCLModel(nn.Module):
    """End-to-end encoder, emotion classifier and contrastive projector."""
    def __init__(self, num_classes, cnn_channels=32, spatial_dim=64,
                 gcn_hidden_dim=64, temporal_dim=128, temporal_layers=2,
                 temporal_heads=4, temporal_ff_dim=256, fusion_dim=128,
                 projection_hidden_dim=128, projection_dim=64,
                 classifier_hidden_dim=64, dropout=0.1, gcn_dropout=0.3,
                 contrastive_temperature=0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.encoder = NSGCLRepresentationEncoder(
            cnn_channels=cnn_channels, spatial_dim=spatial_dim,
            gcn_hidden_dim=gcn_hidden_dim, temporal_dim=temporal_dim,
            temporal_layers=temporal_layers, temporal_heads=temporal_heads,
            temporal_ff_dim=temporal_ff_dim, fusion_dim=fusion_dim,
            dropout=dropout, gcn_dropout=gcn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes))
        self.projector = ContrastiveProjectionHead(
            input_dim=fusion_dim, hidden_dim=projection_hidden_dim,
            projection_dim=projection_dim, dropout=dropout)
        self.contrastive_loss = InfoNCELoss(temperature=contrastive_temperature)
        self.num_classes = int(num_classes)

    def forward(self, time_frequency, adjacency, padding_mask=None):
        encoded = self.encoder(time_frequency, adjacency, padding_mask)
        result = dict(encoded)
        result["logits"] = self.classifier(encoded["clip_embedding"])
        result["probabilities"] = F.softmax(result["logits"], dim=-1)
        result["projection"] = self.projector(encoded["clip_embedding"])
        return result

    def loss(self, output_view1, output_view2, labels,
             lambda_contrastive=0.1, neuro_symbolic_loss=None,
             lambda_neuro_symbolic=0.1):
        """Combine two-view classification, InfoNCE and optional NS penalty."""
        if lambda_contrastive < 0 or lambda_neuro_symbolic < 0:
            raise ValueError("Loss weights must be non-negative.")
        device = output_view1["logits"].device
        labels = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
        logits1, logits2 = output_view1["logits"], output_view2["logits"]
        if logits1.shape != logits2.shape or logits1.shape[0] != labels.numel():
            raise ValueError("View logits and labels must have matching batch sizes.")
        classification = 0.5 * (
            F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))
        contrastive = self.contrastive_loss(
            output_view1["projection"], output_view2["projection"])
        if neuro_symbolic_loss is None:
            neuro_symbolic = classification.new_zeros(())
        else:
            if neuro_symbolic_loss.ndim != 0:
                raise ValueError("neuro_symbolic_loss must be a scalar tensor.")
            neuro_symbolic = neuro_symbolic_loss
        total = (classification + lambda_contrastive * contrastive +
                 lambda_neuro_symbolic * neuro_symbolic)
        return {"loss": total, "classification_loss": classification,
                "contrastive_loss": contrastive,
                "neuro_symbolic_loss": neuro_symbolic}


def compute_neuro_symbolic_loss(regularizer, probabilities, faa, beta_plv,
                                positive_class_index, negative_class_index,
                                high_arousal_probability):
    """Apply explicit dataset-specific class mappings to the NS regularizer."""
    n_classes = probabilities.shape[-1]
    if not (0 <= positive_class_index < n_classes):
        raise ValueError("positive_class_index is outside the class range.")
    if not (0 <= negative_class_index < n_classes):
        raise ValueError("negative_class_index is outside the class range.")
    return regularizer(
        faa=faa, beta_plv=beta_plv,
        positive_probability=probabilities[:, positive_class_index],
        negative_probability=probabilities[:, negative_class_index],
        high_arousal_probability=high_arousal_probability)["loss"]
