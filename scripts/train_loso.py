"""Run subject-wise LOSO training on a prepared NPZ feature archive.

Example:
python -m scripts.train_loso --data data/processed/deap_features.npz --config configs/loso_deap.yaml
"""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.feature_archive import load_feature_archive, EEGFeatureDataset
from data.contrastive_pairs import SameSampleTwoViewDataset
from models.nsgcl import NSGCLModel
from training.augmentations import EEGTimeFrequencyAugment
from training.loso import make_loso_folds, FeatureStandardizer
from training.metrics import classification_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["time_frequency"].to(device)
            adjacency = batch["adjacency"].to(device)
            output = model(x, adjacency)
            predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
            truths.extend(batch["label"].cpu().tolist())
    return np.asarray(truths), np.asarray(predictions)


def train_fold(archive, fold, config, output_dir, device, num_classes):
    scaler = FeatureStandardizer().fit(archive["features"][fold.scaler_fit_indices])
    scaled_features = scaler.transform(archive["features"])

    train_base = EEGFeatureDataset(archive, fold.train_indices, scaled_features)
    augmentation = EEGTimeFrequencyAugment(**config.get("augmentation", {}))
    train_dataset = SameSampleTwoViewDataset(
        train_base, transform=augmentation, eeg_key="time_frequency")
    training_config = config["training"]
    if len(train_dataset) < 2:
        raise ValueError("At least two training samples are needed for InfoNCE.")
    batch_size = min(int(training_config["batch_size"]), len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=int(training_config.get("num_workers", 0)))

    val_dataset = EEGFeatureDataset(archive, fold.validation_indices, scaled_features)
    test_dataset = EEGFeatureDataset(archive, fold.test_indices, scaled_features)
    eval_batch_size = int(training_config.get("eval_batch_size", 128))
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    model = NSGCLModel(num_classes=num_classes, **dict(config.get("model", {}))).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 1e-4)))

    fold_dir = Path(output_dir) / f"heldout_{str(fold.held_out_subject)}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history_file = fold_dir / "training_history.csv"
    best_score, best_epoch, stale_epochs = -float("inf"), 0, 0

    with history_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "epoch", "train_loss", "classification_loss", "contrastive_loss",
            "validation_accuracy", "validation_macro_f1"])
        writer.writeheader()
        for epoch in range(1, int(training_config["max_epochs"]) + 1):
            model.train()
            total_losses, classification_losses, contrastive_losses = [], [], []
            for batch in train_loader:
                view1 = batch["view1"].to(device)
                view2 = batch["view2"].to(device)
                adjacency = batch["adjacency"].to(device)
                labels = batch["label"].to(device)
                # Default DataLoader collation returns subject IDs as a list of strings.
                subject_ids = batch["subject_id"]
                trial_ids = batch["trial_id"].to(device)

                output1 = model(view1, adjacency)
                output2 = model(view2, adjacency)
                losses = model.loss(
                    output1, output2, labels,
                    subject_ids=subject_ids,
                    trial_ids=trial_ids,
                    lambda_contrastive=float(
                        training_config.get("lambda_contrastive", 0.1)))

                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                clip = training_config.get("gradient_clip_norm", 1.0)
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                optimizer.step()
                total_losses.append(float(losses["loss"].detach().cpu()))
                classification_losses.append(float(losses["classification_loss"].detach().cpu()))
                contrastive_losses.append(float(losses["contrastive_loss"].detach().cpu()))

            y_val, pred_val = evaluate(model, val_loader, device)
            val_metrics = classification_metrics(
                y_val, pred_val, labels=np.arange(num_classes))
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(total_losses)),
                "classification_loss": float(np.mean(classification_losses)),
                "contrastive_loss": float(np.mean(contrastive_losses)),
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["f1_macro"],
            }
            writer.writerow(row)
            stream.flush()
            score = val_metrics["f1_macro"]
            if score > best_score:
                best_score, best_epoch, stale_epochs = score, epoch, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_macro_f1": best_score,
                    "held_out_subject": str(fold.held_out_subject),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "config": config,
                }, fold_dir / "best_model.pt")
            else:
                stale_epochs += 1
                if stale_epochs >= int(training_config.get("patience", 15)):
                    break

    checkpoint = torch.load(
        fold_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_test, pred_test = evaluate(model, test_loader, device)
    metrics = classification_metrics(y_test, pred_test, labels=np.arange(num_classes))
    result = {
        "held_out_subject": str(fold.held_out_subject),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_score),
        **{key: value for key, value in metrics.items()
           if key not in ("confusion_matrix", "class_labels")},
        "confusion_matrix": metrics["confusion_matrix"],
        "class_labels": metrics["class_labels"],
        "n_train": int(len(fold.train_indices)),
        "n_validation": int(len(fold.validation_indices)),
        "n_test": int(len(fold.test_indices)),
    }
    with (fold_dir / "test_metrics.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train NS-GCL with subject-wise LOSO.")
    parser.add_argument("--data", required=True, help="Prepared NPZ feature archive.")
    parser.add_argument("--config", default="configs/loso_deap.yaml")
    parser.add_argument("--output", default="outputs/loso")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    archive = load_feature_archive(args.data)
    classes = np.unique(archive["labels"])
    if not np.array_equal(classes, np.arange(classes.size)) or classes.size < 2:
        raise ValueError("Class IDs must be contiguous integers starting at zero (at least 2).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = make_loso_folds(
        archive["subject_ids"],
        validation_subject_count=int(
            config["training"].get("validation_subject_count", 1)),
        random_state=seed)
    results = []
    for fold in folds:
        print(f"Training fold with held-out subject: {fold.held_out_subject}")
        results.append(train_fold(
            archive, fold, config, output_dir, device, int(classes.size)))

    csv_fields = [
        "held_out_subject", "best_epoch", "best_validation_macro_f1",
        "accuracy", "balanced_accuracy", "precision_macro",
        "recall_macro", "f1_macro", "n_train", "n_validation", "n_test"]
    with (output_dir / "loso_subject_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in csv_fields})
    summary = {"subjects": len(results), "metrics": {}}
    for key in ("accuracy", "balanced_accuracy", "precision_macro",
                "recall_macro", "f1_macro"):
        values = np.asarray([result[key] for result in results], dtype=float)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    with (output_dir / "loso_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
