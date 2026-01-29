import os
import sys
import cv2
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import torchvision
from torchvision.ops import box_iou, generalized_box_iou

from einops import rearrange

warnings.filterwarnings("ignore")


class Config:
    NUM_CLASSES = 2
    CLASS_NAMES = ["fight", "collapse", "no_object"]
    NUM_FRAMES = 16
    IMG_SIZE = 224
    NUM_QUERIES = 12
    HIDDEN_DIM = 256
    NUM_DECODER_LAYERS = 4
    NUM_HEADS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = True
    SCORE_THRESHOLD = 0.3
    MODEL_PATH = "best_model.pth"


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0, 1)
    y1 = (cy - 0.5 * h).clamp(0, 1)
    x2 = (cx + 0.5 * w).clamp(0, 1)
    y2 = (cy + 0.5 * h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


class VideoBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        backbone = torchvision.models.video.r2plus1d_18(pretrained=False)
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


@torch.no_grad()
def extract_predictions(pred_logits, pred_boxes, score_threshold):
    probs = pred_logits.softmax(dim=-1)
    predictions = []
    
    for query_idx in range(pred_logits.shape[0]):
        fg_probs = probs[query_idx, :-1]
        max_prob, class_idx = fg_probs.max(dim=0)
        
        if max_prob >= score_threshold:
            class_id = int(class_idx.cpu())
            score = float(max_prob.cpu())
            box = pred_boxes[query_idx].cpu().numpy()
            box_xyxy = box_cxcywh_to_xyxy(torch.from_numpy(box).unsqueeze(0)).squeeze(0).numpy()
            
            predictions.append({
                "class": class_id,
                "class_name": Config.CLASS_NAMES[class_id],
                "score": score,
                "box_xyxy": box_xyxy
            })
    
    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions


def test_videos(input_dir, output_file, model_path):
    print("====== Testing ======")
    print(f"Device: {Config.DEVICE}")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Model path: {model_path}")
    print(f"Score threshold: {Config.SCORE_THRESHOLD}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if len(video_files) == 0:
        print(f"Error: No video files found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} video files")
    
    model = STAR().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    results = []
    processed = 0
    failed = 0
    
    for video_file in video_files:
        try:
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing [{processed+1}/{len(video_files)}]: {video_file}")
            
            frames = read_video_cv2(video_path, Config.NUM_FRAMES)
            frames = frames.unsqueeze(0).to(Config.DEVICE)
            
            with autocast(enabled=(Config.AMP and Config.DEVICE == "cuda")):
                outputs = model(frames)
            
            predictions = extract_predictions(
                outputs["pred_logits"][0],
                outputs["pred_boxes"][0],
                Config.SCORE_THRESHOLD
            )
            
            row = {"video_name": video_file}
            
            if len(predictions) == 0:
                row["pred_class_1"] = Config.CLASS_NAMES[-1]
                row["x1_1"] = 0.0
                row["y1_1"] = 0.0
                row["x2_1"] = 0.0
                row["y2_1"] = 0.0
                print(f"  → Prediction: {Config.CLASS_NAMES[-1]} (no detections above threshold)")
            else:
                for i, pred in enumerate(predictions, 1):
                    row[f"pred_class_{i}"] = pred["class_name"]
                    row[f"x1_{i}"] = float(pred["box_xyxy"][0])
                    row[f"y1_{i}"] = float(pred["box_xyxy"][1])
                    row[f"x2_{i}"] = float(pred["box_xyxy"][2])
                    row[f"y2_{i}"] = float(pred["box_xyxy"][3])
                
                print(f"  → Predictions: {len(predictions)} detections")
                for i, pred in enumerate(predictions, 1):
                    print(f"     {i}. {pred['class_name']} (score: {pred['score']:.3f})")
            
            results.append(row)
            processed += 1
            
        except Exception as e:
            print(f"  → Error processing {video_file}: {str(e)}")
            failed += 1
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n====== Summary ======")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test STAR model on video clips")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing video clips")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Score threshold for predictions")
    
    args = parser.parse_args()
    
    Config.SCORE_THRESHOLD = args.score_threshold
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_log_{timestamp}.txt"
    
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}\n")
        
        test_videos(args.input_dir, args.output_file, args.model_path)
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.close()


if __name__ == "__main__":
    main()
