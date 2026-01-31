# STAR: Spatio-Temporal Action Recognition

STAR is an end-to-end transformer-based model for spatio-temporal action detection in videos.  
It is inspired by DETR-style set prediction and uses an R(2+1)D backbone for spatio-temporal feature extraction.

The model detects WHAT action is occurring and WHERE it occurs in the video frame, without requiring external proposals or non-maximum suppression.

--------------------------------------------------------------------------------
KEY FEATURES
--------------------------------------------------------------------------------

• End-to-end action detection (no proposals, no NMS)
• DETR-style set-based prediction with Hungarian matching
• Spatio-temporal understanding using 3D CNN + Transformer
• Mixed precision training with AMP
• Lightweight model (~40M parameters, ~155MB)
• Runs on CPU or GPU
• Simple inference with bounding box visualization
• Easily adaptable to custom action categories

--------------------------------------------------------------------------------
WHAT THIS MODEL DOES
--------------------------------------------------------------------------------

Given a video clip, the model:

• Classifies the action occurring in the video
• Localizes the action spatially using bounding boxes
• Processes the video end-to-end
• Outputs predictions with confidence scores
• Generates annotated videos with visualizations

Use cases include:
• Action recognition
• Event detection
• Video understanding
• Automated video analysis
• Behavior monitoring

--------------------------------------------------------------------------------
PRETRAINED MODEL WEIGHTS
--------------------------------------------------------------------------------

Pretrained weights are available on Hugging Face:

Project Page:
https://huggingface.co/spaces/ayushsaun/video-action-transformer

Download via terminal:

wget https://huggingface.co/spaces/ayushsaun/video-action-transformer/resolve/main/best_model.pth

Download via Hugging Face Hub (Python):

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="ayushsaun/video-action-transformer",
    filename="best_model.pth"
)

--------------------------------------------------------------------------------
MODEL OVERVIEW
--------------------------------------------------------------------------------

Backbone: R(2+1)D-18 (pretrained on Kinetics-400)
Architecture: DETR-style Transformer
Total Parameters: ~40.5M
Model Size: ~155 MB
Frames per Video: 16
Hidden Dimension: 256
Number of Queries: 12
Precision: FP16 / FP32 (AMP)

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

video-action-transformer/
├── main.py          Training script
├── test.py          Inference and visualization
├── requirements.txt Python dependencies
├── Report.pdf       Detailed project report
└── README.md        Documentation

Note: Model weights (best_model.pth) are hosted separately on Hugging Face.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

• Python 3.8 or higher
• PyTorch
• CUDA-capable GPU recommended
• 4GB+ GPU memory for training
• 2GB+ GPU memory for inference
• ~1GB disk space

CPU-only inference is supported (slower).

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

Clone the repository:

git clone https://github.com/ayushsaun24024/video-action-transformer.git
cd video-action-transformer

Install dependencies:

pip install -r requirements.txt

Download model weights:

wget https://huggingface.co/spaces/ayushsaun/video-action-transformer/resolve/main/best_model.pth

Optional virtual environment:

python3 -m venv star_env
source star_env/bin/activate        (Linux/Mac)
star_env\Scripts\activate           (Windows)

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

INFERENCE (Action Detection on Videos):

python3 test.py \
  --input_dir ./test_videos \
  --output_file predictions.csv \
  --model_path best_model.pth

Outputs:
• predictions.csv – detection results
• annotated_videos/ – videos with bounding boxes
• test_log_*.txt – execution logs

--------------------------------------------------------------------------------
TRAINING ON CUSTOM DATASET
--------------------------------------------------------------------------------

python3 main.py \
  --input_dir /path/to/dataset \
  --output_dir ./output

--------------------------------------------------------------------------------
DATASET FORMAT
--------------------------------------------------------------------------------

Training Dataset Structure:

dataset/
├── action_challenge.csv
├── video1.mp4
├── video2.mp4
├── label1.txt
└── label2.txt

CSV format (action_challenge.csv):

video_file,label_file
video1.mp4,label1.txt
video2.mp4,label2.txt

Label file format (.txt):

<class_id> <x1> <y1> <x2> <y2>

Example:
0 150 200 450 600

• Coordinates are pixel-based (xyxy)
• Class IDs are automatically mapped during training

--------------------------------------------------------------------------------
INFERENCE DATASET
--------------------------------------------------------------------------------

test_videos/
├── video1.mp4
├── video2.avi
└── video3.mov

Supported formats:
.mp4, .avi, .mov, .mkv

--------------------------------------------------------------------------------
DEFAULT TRAINING CONFIGURATION
--------------------------------------------------------------------------------

Epochs: 50
Batch Size: 2
Learning Rate: 5e-5
Frames per Video: 16
Image Size: 224 x 224
Optimizer: AdamW
Scheduler: Cosine Annealing
Validation Split: 20%

Modify these values in the Config class inside main.py.

--------------------------------------------------------------------------------
INFERENCE DETAILS
--------------------------------------------------------------------------------

python3 test.py \
  --input_dir ./videos \
  --output_file results.csv \
  --score_threshold 0.5

Bounding boxes are normalized (0–1 range).
Multiply by video width and height to get pixel coordinates.

Example output:

video_name,pred_class,x1,y1,x2,y2
video1.mp4,action_0,0.234,0.456,0.678,0.890

--------------------------------------------------------------------------------
MODEL ARCHITECTURE
--------------------------------------------------------------------------------

Backbone:
• R(2+1)D-18
• Factorized 3D convolutions
• Kinetics-400 pretrained

Transformer Decoder:
• 4 layers
• 8 attention heads
• 2048-dimensional feedforward network

Prediction Heads:
• Action classification
• Bounding box regression (cx, cy, w, h)

Training Loss:
• Classification loss
• L1 loss (weighted)
• GIoU loss

Matching is performed using the Hungarian algorithm.

--------------------------------------------------------------------------------
ADVANCED USAGE
--------------------------------------------------------------------------------

Batch inference on multiple directories:

for d in dir1 dir2 dir3; do
  python3 test.py --input_dir $d --output_file ${d}_results.csv
done

Force CPU inference:

CUDA_VISIBLE_DEVICES="" python3 test.py --input_dir ./videos --output_file out.csv

Modify action classes in main.py:

CLASS_NAMES = ["action_A", "action_B", "no_object"]
NUM_CLASSES = 3

--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

CUDA out of memory:
• Set BATCH_SIZE = 1 in main.py

Video read errors:
• Install codecs: sudo apt-get install libavcodec-extra
• Verify video: ffmpeg -i video.mp4 -f null -

Model loading errors:
• Ensure best_model.pth exists
• File size should be ~155MB
• Re-download if corrupted

--------------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------------

Apache License 2.0

This project is open source and licensed under the Apache License, Version 2.0.
You are free to use, modify, and distribute it for both research and commercial purposes.

--------------------------------------------------------------------------------
CONTRIBUTING
--------------------------------------------------------------------------------

Contributions are welcome.

Potential improvements:
• Multi-instance detection
• Temporal localization
• Model compression
• Real-time inference
• Data augmentation strategies

Open an issue or submit a pull request.

--------------------------------------------------------------------------------
ACKNOWLEDGMENTS
--------------------------------------------------------------------------------

• Meta AI Research (DETR)
• Google Research (STAR, R(2+1)D)
• PyTorch Team
• Hugging Face
• Open-source computer vision community
