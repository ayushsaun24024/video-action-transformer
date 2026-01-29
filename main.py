import os
import cv2
import math
import json
import torch
import random
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from einops import rearrange
from torch.optim import AdamW
import torch.nn.functional as F
from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import box_iou, generalized_box_iou
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import classification_report, confusion_matrix


warnings.filterwarnings("ignore")


class Config:
    SEED = 42
    DATASET_PATH = ""
    CSV_PATH = ""
    OUTPUT_DIR = "./output"

    NUM_CLASSES = 2
    CLASS_NAMES = ["fight", "collapse", "no_object"]

    NUM_FRAMES = 16
    IMG_SIZE = 224
    NUM_QUERIES = 12
    HIDDEN_DIM = 256
    NUM_DECODER_LAYERS = 4
    NUM_HEADS = 8

    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    TEST_FREQUENCY = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BOUNDING_BOX_L1_LOSS_WEIGHT = 5.0
    GENERALIZED_IOU_LOSS_WEIGHT = 2.0
    CLASSIFICATION_LOSS_WEIGHT = 2.0
    NO_OBJECT_CLASS_WEIGHT = 0.1

    AMP = True
    GRAD_CLIP_NORM = 0.5

    LABEL_OFFSET = 0
    VALIDATION_SIZE = 0.2
    STRATIFY_MIN_PER_CLASS = 2

    SCORE_THRESHOLD = 0.3
    IOU_TP_THRESHOLD = 0.5

    SAVE_BY = "combo"
    COMBO_ALPHA = 0.2

    SPLIT_JSON = "split.json"
    BEST_CKPT = "best_model.pth"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return videos, targets


def read_video_cv2(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 30

    indices = np.linspace(0, max(1, total_frames - 1), num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
            frames.append(frame)
        else:
            frames.append(frames[-1] if len(frames) else np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8))

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.stack(frames[:num_frames])
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames


def parse_annotation(txt_path: str):
    try:
        with open(txt_path, "r") as f:
            line = f.readline().strip()
    except:
        return None, None
    if not line:
        return None, None
    parts = line.split()
    if len(parts) < 5:
        return None, None
    raw_label = int(float(parts[0]))
    bbox = [float(x) for x in parts[1:5]]
    return raw_label, bbox


def normalize_bbox_xyxy_to_cxcywh(bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = bbox_xyxy
    img_w = max(int(img_w), 1)
    img_h = max(int(img_h), 1)

    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    cx = float(np.clip(cx, 0, 1))
    cy = float(np.clip(cy, 0, 1))
    w = float(np.clip(w, 0, 1))
    h = float(np.clip(h, 0, 1))
    return [cx, cy, w, h]


def infer_label_offset(data_list):
    labels = set()
    for item in data_list:
        if not item["label_file"]:
            continue
        txt_path = os.path.join(Config.DATASET_PATH, item["label_file"])
        raw_label, _ = parse_annotation(txt_path)
        if raw_label is not None:
            labels.add(raw_label)
    if (2 in labels) and (0 not in labels):
        return 1
    return 0


def get_stratify_label(item):
    if not item["label_file"]:
        return Config.NUM_CLASSES
    txt_path = os.path.join(Config.DATASET_PATH, item["label_file"])
    raw_label, _ = parse_annotation(txt_path)
    if raw_label is None:
        return Config.NUM_CLASSES
    mapped = raw_label - Config.LABEL_OFFSET
    if 0 <= mapped < Config.NUM_CLASSES:
        return mapped
    return Config.NUM_CLASSES


def load_full_list():
    df = pd.read_csv(Config.CSV_PATH)
    data_list = []
    for _, row in df.iterrows():
        video_file = row["video_file"]
        label_file = row["label_file"] if pd.notna(row["label_file"]) else None
        data_list.append({"video_file": video_file, "label_file": label_file})
    random.Random(Config.SEED).shuffle(data_list)
    return data_list


def make_and_save_split():
    data_list = load_full_list()
    Config.LABEL_OFFSET = infer_label_offset(data_list)

    stratify_y = [get_stratify_label(item) for item in data_list]
    counts = pd.Series(stratify_y).value_counts().to_dict()
    can_stratify = all(v >= Config.STRATIFY_MIN_PER_CLASS for v in counts.values())

    if can_stratify:
        train_list, val_list = train_test_split(
            data_list,
            test_size=Config.VALIDATION_SIZE,
            random_state=Config.SEED,
            stratify=stratify_y
        )
    else:
        train_list, val_list = train_test_split(
            data_list,
            test_size=Config.VALIDATION_SIZE,
            random_state=Config.SEED,
            shuffle=True
        )

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    split_path = os.path.join(Config.OUTPUT_DIR, Config.SPLIT_JSON)
    with open(split_path, "w") as f:
        json.dump(
            {
                "label_offset": Config.LABEL_OFFSET,
                "train": train_list,
                "val": val_list,
                "counts": {str(k): int(v) for k, v in counts.items()},
                "stratified": bool(can_stratify),
                "seed": Config.SEED,
            },
            f,
            indent=2,
        )
    return list(train_list), list(val_list), counts, can_stratify


def load_saved_split():
    split_path = os.path.join(Config.OUTPUT_DIR, Config.SPLIT_JSON)
    if not os.path.exists(split_path):
        return make_and_save_split()

    with open(split_path, "r") as f:
        obj = json.load(f)
    Config.LABEL_OFFSET = int(obj["label_offset"])
    counts = {int(k): int(v) for k, v in obj.get("counts", {}).items()}
    return obj["train"], obj["val"], counts, bool(obj.get("stratified", False))


def format_split_counts(counts_dict):
    def name_for(k):
        if k == Config.NUM_CLASSES:
            return "no_object"
        return Config.CLASS_NAMES[k]
    keys = sorted(counts_dict.keys())
    return ", ".join([f"{name_for(k)}: {counts_dict[k]}" for k in keys])


class ActionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_path = os.path.join(Config.DATASET_PATH, item["video_file"])
        frames = read_video_cv2(video_path, Config.NUM_FRAMES)

        labels = torch.zeros((0,), dtype=torch.long)
        boxes = torch.zeros((0, 4), dtype=torch.float32)

        if item["label_file"]:
            txt_path = os.path.join(Config.DATASET_PATH, item["label_file"])
            raw_label, bbox_xyxy = parse_annotation(txt_path)
            if (raw_label is not None) and (bbox_xyxy is not None):
                mapped = raw_label - Config.LABEL_OFFSET
                if 0 <= mapped < Config.NUM_CLASSES:
                    cap = cv2.VideoCapture(video_path)
                    img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()

                    bbox_norm = normalize_bbox_xyxy_to_cxcywh(bbox_xyxy, img_w, img_h)
                    labels = torch.tensor([mapped], dtype=torch.long)
                    boxes = torch.tensor([bbox_norm], dtype=torch.float32)

        return frames, {"labels": labels, "boxes": boxes}


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset: ActionDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.pos_indices = [i for i, item in enumerate(dataset.data_list) if item["label_file"] is not None]
        self.all_indices = list(range(len(dataset)))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + random.randint(0, 10_000))
        all_idx = np.array(self.all_indices, dtype=int)
        rng.shuffle(all_idx)

        if len(self.pos_indices) == 0:
            for i in range(0, len(all_idx), self.batch_size):
                yield all_idx[i:i + self.batch_size].tolist()
            return

        pos_idx = np.array(self.pos_indices, dtype=int)
        rng.shuffle(pos_idx)
        pos_ptr = 0

        for i in range(0, len(all_idx), self.batch_size):
            batch = all_idx[i:i + self.batch_size].tolist()
            if len(batch) == 0:
                continue
            has_positive = any(self.dataset.data_list[j]["label_file"] is not None for j in batch)
            if not has_positive:
                if pos_ptr >= len(pos_idx):
                    rng.shuffle(pos_idx)
                    pos_ptr = 0
                batch[0] = int(pos_idx[pos_ptr])
                pos_ptr += 1
            yield batch


class VideoBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        backbone = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.stem = nn.Sequential(backbone.stem, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.proj = nn.Conv3d(512, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.proj(x)
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = self.norm(x)
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        b, n, _ = x.shape
        if n >= self.embedding.num_embeddings:
            raise ValueError(f"Sequence length {n} exceeds max_len {self.embedding.num_embeddings}")
        pos = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.embedding(pos)


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, queries, memory):
        return self.decoder(queries, memory)


class DetectionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes + 1)
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4), nn.Sigmoid()
        )

    def forward(self, x):
        return self.class_head(x), self.bbox_head(x)


class STAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = VideoBackbone(Config.HIDDEN_DIM)
        self.pos_embedding = PositionEmbedding(Config.HIDDEN_DIM, max_len=10000)
        self.queries = nn.Embedding(Config.NUM_QUERIES, Config.HIDDEN_DIM)
        self.decoder = TransformerDecoder(Config.HIDDEN_DIM, Config.NUM_HEADS, Config.NUM_DECODER_LAYERS)
        self.head = DetectionHead(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, x):
        memory = self.pos_embedding(self.backbone(x))
        b = x.shape[0]
        queries = self.queries.weight.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(queries, memory)
        logits, boxes = self.head(decoded)
        return {"pred_logits": logits, "pred_boxes": boxes}


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0, 1)
    y1 = (cy - 0.5 * h).clamp(0, 1)
    x2 = (cx + 0.5 * w).clamp(0, 1)
    y2 = (cy + 0.5 * h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def hungarian_matching(outputs, targets):
    bs, _ = outputs["pred_logits"].shape[:2]
    device = outputs["pred_logits"].device

    indices = []
    out_prob = outputs["pred_logits"].softmax(-1)
    out_bbox = outputs["pred_boxes"]

    for i in range(bs):
        tgt_ids = targets[i]["labels"].to(device)
        tgt_bbox = targets[i]["boxes"].to(device)

        if tgt_ids.numel() == 0:
            indices.append(
                (torch.empty((0,), dtype=torch.int64, device=device),
                 torch.empty((0,), dtype=torch.int64, device=device))
            )
            continue

        cost_class = -out_prob[i, :, tgt_ids]
        cost_bbox = torch.cdist(out_bbox[i], tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[i]), box_cxcywh_to_xyxy(tgt_bbox))

        C = (Config.CLASSIFICATION_LOSS_WEIGHT * cost_class
             + Config.BOUNDING_BOX_L1_LOSS_WEIGHT * cost_bbox
             + Config.GENERALIZED_IOU_LOSS_WEIGHT * cost_giou)

        src_idx, tgt_idx = linear_sum_assignment(C.detach().cpu().numpy())
        indices.append(
            (torch.as_tensor(src_idx, dtype=torch.int64, device=device),
             torch.as_tensor(tgt_idx, dtype=torch.int64, device=device))
        )

    return indices


def compute_loss(outputs, targets, indices):
    device = outputs["pred_logits"].device
    bs, num_queries, num_logits = outputs["pred_logits"].shape
    expected_logits = Config.NUM_CLASSES + 1
    if num_logits != expected_logits:
        raise ValueError(f"Model output has {num_logits} classes, expected {expected_logits}")

    target_classes = torch.full((bs, num_queries), Config.NUM_CLASSES, dtype=torch.int64, device=device)
    for i, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() > 0:
            target_classes[i, src_idx] = targets[i]["labels"][tgt_idx].to(device)

    class_weights = torch.ones(Config.NUM_CLASSES + 1, device=device)
    class_weights[-1] = Config.NO_OBJECT_CLASS_WEIGHT

    loss_cls = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes, weight=class_weights)

    src_boxes_list, tgt_boxes_list = [], []
    for i, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() > 0:
            src_boxes_list.append(outputs["pred_boxes"][i, src_idx])
            tgt_boxes_list.append(targets[i]["boxes"][tgt_idx].to(device))

    if len(src_boxes_list) > 0:
        src_boxes = torch.cat(src_boxes_list, dim=0)
        tgt_boxes = torch.cat(tgt_boxes_list, dim=0)

        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1.0 - torch.diag(giou)).mean()

        diag_iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        matched_boxes = int(diag_iou.numel())
        matched_iou_sum = float(diag_iou.sum().detach().cpu())
    else:
        loss_l1 = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        matched_boxes = 0
        matched_iou_sum = 0.0

    total = (Config.CLASSIFICATION_LOSS_WEIGHT * loss_cls
             + Config.BOUNDING_BOX_L1_LOSS_WEIGHT * loss_l1
             + Config.GENERALIZED_IOU_LOSS_WEIGHT * loss_giou)

    metrics = {
        "classification_loss": float(loss_cls.detach().cpu()),
        "bounding_box_l1_loss": float(loss_l1.detach().cpu()),
        "generalized_iou_loss": float(loss_giou.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "matched_boxes": matched_boxes,
        "matched_iou_sum": matched_iou_sum,
    }
    return total, metrics


@torch.no_grad()
def select_video_level_prediction_and_query(pred_logits, score_threshold: float):
    probs = pred_logits.softmax(dim=-1)
    fg = probs[:, :-1]
    best_score_per_class, best_query_per_class = fg.max(dim=0)
    best_class = int(best_score_per_class.argmax().cpu())
    best_score = float(best_score_per_class[best_class].cpu())
    best_query = int(best_query_per_class[best_class].cpu())

    if best_score < score_threshold:
        return Config.NUM_CLASSES, None, best_score
    return best_class, best_query, best_score


@torch.no_grad()
def det_acc_iou_threshold(outputs, targets, score_threshold: float, iou_threshold: float):
    device = outputs["pred_logits"].device
    bs = outputs["pred_logits"].shape[0]
    correct, total = 0, 0

    for i in range(bs):
        pred_label, pred_q, _ = select_video_level_prediction_and_query(outputs["pred_logits"][i], score_threshold)
        true_label = int(targets[i]["labels"][0].cpu()) if targets[i]["labels"].numel() > 0 else Config.NUM_CLASSES
        has_box = bool(targets[i]["boxes"].numel() > 0)

        total += 1

        if not has_box:
            correct += int(pred_label == Config.NUM_CLASSES)
            continue

        if pred_label != true_label or pred_q is None:
            continue

        pred_box = outputs["pred_boxes"][i, pred_q].unsqueeze(0)
        gt_box = targets[i]["boxes"][0].unsqueeze(0).to(device)

        iou = float(box_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(gt_box))[0, 0].cpu())
        correct += int(iou >= iou_threshold)

    return correct, total


def print_block(title, metrics):
    print(f"==== {title} ====")
    print(
        f"Loss: {metrics['total_loss']:.4f} | "
        f"Cls: {metrics['classification_loss']:.4f} | "
        f"L1: {metrics['bounding_box_l1_loss']:.4f} | "
        f"GIoU: {metrics['generalized_iou_loss']:.4f}"
    )
    msg = (
        f"Video-acc: {metrics['video_level_accuracy']:.4f} | "
        f"Mean IoU(matched): {metrics['mean_iou_on_matched_boxes']:.4f} | "
        f"Matched: {int(metrics['matched_boxes_total'])}"
    )
    if "detection_accuracy_iou50" in metrics:
        msg += f" | Det-acc@0.5: {metrics['detection_accuracy_iou50']:.4f}"
    print(msg)


def train_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss_value = 0.0
    metric_sums = defaultdict(float)
    matched_boxes_total = 0
    matched_iou_sum_total = 0.0
    correct, total = 0, 0

    for videos, targets in dataloader:
        videos = videos.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(Config.AMP and device == "cuda")):
            outputs = model(videos)
            indices = hungarian_matching(outputs, targets)
            loss, m = compute_loss(outputs, targets, indices)

        if Config.AMP and device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

        total_loss_value += float(loss.detach().cpu())
        metric_sums["classification_loss"] += m["classification_loss"]
        metric_sums["bounding_box_l1_loss"] += m["bounding_box_l1_loss"]
        metric_sums["generalized_iou_loss"] += m["generalized_iou_loss"]
        matched_boxes_total += int(m["matched_boxes"])
        matched_iou_sum_total += float(m["matched_iou_sum"])

        for i, t in enumerate(targets):
            pred_label, _, _ = select_video_level_prediction_and_query(outputs["pred_logits"][i], Config.SCORE_THRESHOLD)
            true_label = int(t["labels"][0].cpu()) if t["labels"].numel() > 0 else Config.NUM_CLASSES
            correct += int(pred_label == true_label)
            total += 1

    batches = max(len(dataloader), 1)
    avg_loss = total_loss_value / batches
    mean_iou_matched = (matched_iou_sum_total / max(matched_boxes_total, 1)) if matched_boxes_total > 0 else 0.0

    metrics = {
        "classification_loss": metric_sums["classification_loss"] / batches,
        "bounding_box_l1_loss": metric_sums["bounding_box_l1_loss"] / batches,
        "generalized_iou_loss": metric_sums["generalized_iou_loss"] / batches,
        "total_loss": avg_loss,
        "video_level_accuracy": correct / max(total, 1),
        "mean_iou_on_matched_boxes": mean_iou_matched,
        "matched_boxes_total": matched_boxes_total,
    }
    return avg_loss, metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss_value = 0.0
    metric_sums = defaultdict(float)
    matched_boxes_total = 0
    matched_iou_sum_total = 0.0

    correct, total = 0, 0
    all_preds, all_labels = [], []

    det_correct_total, det_total_total = 0, 0

    for videos, targets in dataloader:
        videos = videos.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=(Config.AMP and device == "cuda")):
            outputs = model(videos)
            indices = hungarian_matching(outputs, targets)
            loss, m = compute_loss(outputs, targets, indices)

        total_loss_value += float(loss.detach().cpu())
        metric_sums["classification_loss"] += m["classification_loss"]
        metric_sums["bounding_box_l1_loss"] += m["bounding_box_l1_loss"]
        metric_sums["generalized_iou_loss"] += m["generalized_iou_loss"]
        matched_boxes_total += int(m["matched_boxes"])
        matched_iou_sum_total += float(m["matched_iou_sum"])

        for i, t in enumerate(targets):
            pred_label, _, _ = select_video_level_prediction_and_query(outputs["pred_logits"][i], Config.SCORE_THRESHOLD)
            true_label = int(t["labels"][0].cpu()) if t["labels"].numel() > 0 else Config.NUM_CLASSES
            correct += int(pred_label == true_label)
            total += 1
            all_preds.append(pred_label)
            all_labels.append(true_label)

        dc, dt = det_acc_iou_threshold(outputs, targets, Config.SCORE_THRESHOLD, Config.IOU_TP_THRESHOLD)
        det_correct_total += dc
        det_total_total += dt

    batches = max(len(dataloader), 1)
    avg_loss = total_loss_value / batches
    mean_iou_matched = (matched_iou_sum_total / max(matched_boxes_total, 1)) if matched_boxes_total > 0 else 0.0
    det_acc = det_correct_total / max(det_total_total, 1)

    metrics = {
        "classification_loss": metric_sums["classification_loss"] / batches,
        "bounding_box_l1_loss": metric_sums["bounding_box_l1_loss"] / batches,
        "generalized_iou_loss": metric_sums["generalized_iou_loss"] / batches,
        "total_loss": avg_loss,
        "video_level_accuracy": correct / max(total, 1),
        "mean_iou_on_matched_boxes": mean_iou_matched,
        "matched_boxes_total": matched_boxes_total,
        "detection_accuracy_iou50": det_acc,
    }
    return avg_loss, metrics, all_preds, all_labels


def score_for_saving(val_metrics):
    if Config.SAVE_BY == "video_acc":
        return float(val_metrics["video_level_accuracy"])
    return float(val_metrics["video_level_accuracy"]) + Config.COMBO_ALPHA * float(val_metrics["detection_accuracy_iou50"])


def train_model():
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    train_list, val_list, counts, stratified = make_and_save_split()

    print(f"Device: {Config.DEVICE}")
    print(f"Dataset path: {Config.DATASET_PATH}")
    print(f"Inferred label offset: {Config.LABEL_OFFSET}")
    print(f"Full dataset class counts: {format_split_counts(counts)}")
    print(f"Split strategy: {'stratified' if stratified else 'random'}")
    print(f"Training samples: {len(train_list)} | Validation samples: {len(val_list)}")

    train_dataset = ActionDataset(train_list)
    val_dataset = ActionDataset(val_list)

    train_batch_sampler = BalancedBatchSampler(train_dataset, batch_size=Config.BATCH_SIZE, seed=Config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = STAR().to(Config.DEVICE)
    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(Config.AMP and Config.DEVICE == "cuda"))

    results = []
    best_score = -1e9

    for epoch in range(Config.NUM_EPOCHS):
        epoch_index = epoch + 1
        lr = float(optimizer.param_groups[0]["lr"])

        _, train_metrics = train_epoch(model, train_loader, optimizer, scaler, Config.DEVICE)
        print(f"\nEpoch {epoch_index}/{Config.NUM_EPOCHS} | LR: {lr:.8f}")
        print_block("Train", train_metrics)

        row = {"epoch": epoch_index, "learning_rate": lr}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})

        if (epoch_index % Config.TEST_FREQUENCY == 0) or (epoch_index == Config.NUM_EPOCHS):
            _, val_metrics, _, _ = evaluate_epoch(model, val_loader, Config.DEVICE)
            print_block("Validation", val_metrics)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            s = score_for_saving(val_metrics)
            if s > best_score:
                best_score = s
                torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, Config.BEST_CKPT))
                print(f"Saved new best model. Score={best_score:.4f} (mode={Config.SAVE_BY})")

        results.append(row)
        scheduler.step()

    pd.DataFrame(results).to_csv(os.path.join(Config.OUTPUT_DIR, "training_results.csv"), index=False)
    print(f"\nTraining complete. Saved: {os.path.join(Config.OUTPUT_DIR, 'training_results.csv')}")
    return model


def test_model():
    set_seed(Config.SEED)

    train_list, val_list, counts, stratified = load_saved_split()

    val_loader = DataLoader(
        ActionDataset(val_list),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = STAR().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, Config.BEST_CKPT)
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))

    _, val_metrics, preds, labels = evaluate_epoch(model, val_loader, Config.DEVICE)

    print("\nFinal evaluation")
    print_block("Validation", val_metrics)

    print("\nClassification report")
    print(classification_report(labels, preds, target_names=Config.CLASS_NAMES, zero_division=0))

    print("\nConfusion matrix")
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    print(cm)

    pd.DataFrame({"true_label": labels, "pred_label": preds}).to_csv(
        os.path.join(Config.OUTPUT_DIR, "test_predictions.csv"), index=False
    )
    print(f"\nSaved: {os.path.join(Config.OUTPUT_DIR, 'test_predictions.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Train STAR model for action detection")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to output directory")
    
    args = parser.parse_args()
    
    Config.DATASET_PATH = args.input_dir
    Config.CSV_PATH = os.path.join(args.input_dir, "action_challenge.csv")
    Config.OUTPUT_DIR = args.output_dir
    
    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Dataset directory not found: {Config.DATASET_PATH}")
        return
    
    if not os.path.exists(Config.CSV_PATH):
        print(f"Error: CSV file not found: {Config.CSV_PATH}")
        return
    
    train_model()
    test_model()
    print(f"\nOutputs directory: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
