import json, logging, os, uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES: List[str] = [
    "a surveillance video of {}",
    "a cctv footage showing {}",
    "a person doing {}",
    "{} in surveillance camera",
]


class LinearAdapterHead(torch.nn.Module):
    """Linear adapter head for feature classification."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if in_features <= 0 or num_classes <= 0:
            raise ValueError(f"in_features ({in_features}) and num_classes ({num_classes}) must be positive")
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.linear.in_features:
            raise ValueError(f"Expected 2D input with {self.linear.in_features} features, got shape {getattr(x, 'shape', None)}")
        return self.linear(x)


def predict(head: LinearAdapterHead, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute predicted class indices and softmax probabilities."""
    head.eval()
    with torch.no_grad():
        logits = head(features)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    return preds, probs


def pool_video_embedding(frame_embeddings: torch.Tensor, mode: str = "mean", top_k: int = 5) -> torch.Tensor:
    """Pool frame-level embeddings into single L2-normalized video embedding."""
    if not isinstance(frame_embeddings, torch.Tensor) or frame_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D frame_embeddings tensor (T, D), got {getattr(frame_embeddings, 'shape', None)}")
    if frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings cannot be empty")
    if frame_embeddings.shape[1] == 0:
        raise ValueError("frame_embeddings feature dimension cannot be zero")

    if mode == "mean":
        return F.normalize(frame_embeddings.mean(dim=0), p=2, dim=-1)
    if mode == "topk_mean":
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = min(top_k, frame_embeddings.shape[0])
        mean_vec = F.normalize(frame_embeddings.mean(dim=0, keepdim=True), p=2, dim=-1)
        sims = (F.normalize(frame_embeddings, p=2, dim=-1) @ mean_vec.T).squeeze(-1)
        top_idx = torch.topk(sims, k=k).indices
        return F.normalize(frame_embeddings[top_idx].mean(dim=0), p=2, dim=-1)
    raise ValueError(f"Unknown pooling mode '{mode}'. Choose 'mean' or 'topk_mean'.")


def encode_prompt_ensemble(labels: List[str], engine: Any, templates: List[str] = DEFAULT_TEMPLATES) -> torch.Tensor:
    """Encode labels across prompt templates, average, and L2-normalize."""
    if not labels:
        dim = getattr(getattr(engine, "model", None), "config", None)
        feat_dim = getattr(getattr(dim, "text_config", None), "hidden_size", 768)
        return torch.empty((0, feat_dim), dtype=torch.float32)

    rows: List[torch.Tensor] = []
    for label in labels:
        clean = label.replace("_", " ").strip().lower()
        prompts = [tmpl.format(clean) for tmpl in templates]
        embeds = engine.encode_texts(prompts)
        if not isinstance(embeds, torch.Tensor):
            embeds = torch.tensor(embeds, dtype=torch.float32)
        embeds = F.normalize(embeds.float(), p=2, dim=-1)
        unit_embed = F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0).cpu().float()
        rows.append(unit_embed)
    return torch.stack(rows, dim=0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute (num_classes, num_classes) confusion matrix using NumPy."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    t = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    if len(t) != len(p):
        raise ValueError(f"y_true ({len(t)}) and y_pred ({len(p)}) length mismatch")
    if len(t) > 0 and ((t < 0).any() or (t >= num_classes).any() or (p < 0).any() or (p >= num_classes).any()):
        raise ValueError(f"Class indices must be in range [0, {num_classes - 1}]")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(t, p):
        cm[true_idx, pred_idx] += 1
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute per-class F1 score (0.0 when TP+FP+FN == 0)."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        denom = 2 * tp + fp + fn
        f1s[c] = (2.0 * tp) / denom if denom > 0 else 0.0
    return f1s


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute macro-averaged F1 score."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return float(np.mean(per_class_f1(y_true, y_pred, num_classes)))


def train_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    weight_decay: float = 1e-2,
    max_epochs: int = 200,
    lr: float = 1e-1,
    patience: int = 10,
    seed: int = 0,
    class_weights: Optional[torch.Tensor] = None,
    init_weight: Optional[torch.Tensor] = None,
) -> Tuple[LinearAdapterHead, Dict[str, Any]]:
    """Train LinearAdapterHead on CPU with early stopping."""
    torch.manual_seed(seed)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D features, got {features.shape}")
    n_samples, in_features = features.shape
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError(f"labels must be 1D tensor with {n_samples} items, got {labels.shape}")

    head = LinearAdapterHead(in_features=in_features, num_classes=num_classes).cpu()
    if init_weight is not None:
        if init_weight.shape != (num_classes, in_features):
            raise ValueError(f"init_weight shape {init_weight.shape} != expected {(num_classes, in_features)}")
        head.linear.weight.data.copy_(init_weight.cpu().float())
        head.linear.bias.data.zero_()

    x_train, y_train = features.cpu().float(), labels.cpu().long()
    has_val = val_features is not None and val_labels is not None
    if has_val:
        x_val, y_val = val_features.cpu().float(), val_labels.cpu().long()
        if x_val.ndim != 2 or x_val.shape[1] != in_features or len(y_val) != x_val.shape[0]:
            raise ValueError("Validation features or labels dimensions mismatch")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.cpu().float() if class_weights is not None else None)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")

    train_losses, val_losses = [], []
    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    patience_cnt = 0
    converged = False
    ep_count = 0

    for ep in range(max_epochs):
        ep_count = ep + 1

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(head(x_train), y_train)
            if weight_decay > 0:
                loss = loss + 0.5 * weight_decay * head.linear.weight.pow(2).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        t_loss = float(loss.item())
        train_losses.append(t_loss)

        if has_val:
            head.eval()
            with torch.no_grad():
                v_loss = float(loss_fn(head(x_val), y_val).item())
            head.train()
            val_losses.append(v_loss)
            eval_metric = v_loss
        else:
            eval_metric = t_loss

        if eval_metric < best_loss - 1e-4:
            best_loss = eval_metric
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                converged = True
                break

    if patience_cnt >= patience or (len(train_losses) >= 2 and abs(train_losses[-1] - train_losses[-2]) < 1e-4):
        converged = True

    head.load_state_dict(best_state)
    return head, {"train_loss": train_losses, "val_loss": val_losses, "epochs": ep_count, "converged": converged}


def save_head(
    path: str,
    head: LinearAdapterHead,
    class_names: List[str],
    hyperparams: Dict[str, Any],
    model_name: str,
    label_map: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Save trained adapter head checkpoint atomically."""
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    weight = head.linear.weight.detach().cpu().numpy()
    bias = head.linear.bias.detach().cpu().numpy()
    num_classes, in_features = weight.shape
    if len(class_names) != num_classes:
        raise ValueError(f"class_names count ({len(class_names)}) != num_classes ({num_classes})")

    meta = {
        "model_name": model_name, "in_features": in_features, "num_classes": num_classes,
        "hyperparams": hyperparams, "label_map": label_map, "metrics": metrics,
    }
    tmp_path = os.path.join(dir_name, f".tmp_{uuid.uuid4().hex}.npz")
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, weight=weight, bias=bias, class_names=np.array(class_names, dtype=str), metadata=json.dumps(meta))
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return path


def load_head(
    path: str,
    expected_model_name: Optional[str] = None,
    expected_classes: Optional[List[str]] = None,
) -> Tuple[LinearAdapterHead, Dict[str, Any]]:
    """Load adapter head checkpoint with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Head checkpoint not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        for k in ("weight", "bias", "class_names", "metadata"):
            if k not in data:
                raise KeyError(f"Corrupt checkpoint, missing key: '{k}'")
        weight = data["weight"]
        bias = data["bias"]
        class_names = [str(c) for c in data["class_names"].tolist()]
        meta = json.loads(str(data["metadata"]))

    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {weight.shape}")
    num_classes, in_features = weight.shape
    if bias.shape != (num_classes,):
        raise ValueError(f"Bias shape {bias.shape} != expected ({num_classes},)")
    if len(class_names) != num_classes:
        raise ValueError(f"class_names count {len(class_names)} != num_classes {num_classes}")

    saved_model = meta.get("model_name")
    if expected_model_name is not None and saved_model != expected_model_name:
        raise ValueError(f"Model mismatch: expected '{expected_model_name}', found '{saved_model}'")
    if expected_classes is not None and list(class_names) != list(expected_classes):
        raise ValueError(f"Classes mismatch: expected {expected_classes}, found {class_names}")

    head = LinearAdapterHead(in_features=in_features, num_classes=num_classes)
    head.linear.weight.data.copy_(torch.from_numpy(weight).float())
    head.linear.bias.data.copy_(torch.from_numpy(bias).float())
    return head, {
        "class_names": class_names, "model_name": saved_model,
        "hyperparams": meta.get("hyperparams", {}), "label_map": meta.get("label_map"),
        "metrics": meta.get("metrics"), "metadata": meta,
    }
