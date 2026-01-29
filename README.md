================================================================================
                STAR: Spatio-Temporal Action Recognition
================================================================================

🚀 End-to-end transformer-based model for detecting actions in video clips 
with spatial localization using DETR-inspired architecture and R(2+1)D backbone.

[Model Weights 🤗]

================================================================================

Python: 3.8+ | PyTorch | License: MIT | Model: 40M params


KEY FEATURES
================================================================================

✨ End-to-End Detection - No external proposals or NMS required
🎯 DETR-Style Architecture - Set-based prediction with Hungarian matching
🎬 Spatio-Temporal Understanding - 3D CNN backbone + Transformer decoder
⚡ Mixed Precision Training - Efficient training with AMP
📦 Lightweight - 155MB model, runs on CPU or GPU
🔧 Easy Deployment - Simple inference API with bounding box visualization
🎯 Generic Action Detection - Adaptable to any action categories


WHAT IT DOES
================================================================================

This model performs spatio-temporal action detection in videos:
    • Identifies WHAT action is occurring (classification)
    • Localizes WHERE in the frame (bounding box)
    • Processes video clips end-to-end
    • Outputs predictions with confidence scores
    • Generates annotated videos with visualizations

Use cases: Action recognition, event detection, video understanding, 
automated video analysis, behavior monitoring


MODEL WEIGHTS
================================================================================

Pre-trained weights available on Hugging Face:

🤗 Download: https://huggingface.co/spaces/ayushsaun/video-action-transformer

Or use the Hugging Face Hub:
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="YOUR_USERNAME/video-action-transformer", 
                                  filename="best_model.pth")


PROJECT STRUCTURE
================================================================================

video-action-transformer/
├── main.py                Training script
├── test.py                Inference script with visualization
├── requirements.txt       Python dependencies
├── Report.pdf             Report of work done in the pipeline
└── README.md             This file

Note: Model weights (best_model.pth) hosted separately on Hugging Face


REQUIREMENTS
================================================================================

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 4GB+ GPU memory for training
- 2GB+ GPU memory for inference
- 1GB disk space for model weights


INSTALLATION
================================================================================

Quick Setup (3 commands):

1. Clone repository:
   git clone https://github.com/ayushsaun24024/video-action-transformer.git
   cd video-action-transformer

2. Install dependencies:
   pip install -r requirements.txt

3. Download model weights:
   wget https://huggingface.co/spaces/ayushsaun/video-action-transformer/resolve/main/best_model.pth

Optional: Create virtual environment
   python3 -m venv star_env
   source star_env/bin/activate  # Linux/Mac
   star_env\Scripts\activate     # Windows


QUICK START
================================================================================

Inference (Detect actions in your videos):

python3 test.py \
  --input_dir ./test_videos \
  --output_file predictions.csv \
  --model_path best_model.pth

Output:
  • predictions.csv - Detection results with bounding boxes
  • annotated_videos/ - Videos with bounding boxes drawn
  • test_log_*.txt - Execution log


Training (Train on your custom dataset):

python3 main.py --input_dir /path/to/dataset --output_dir ./output

See DATASET STRUCTURE section below for data format requirements.


DATASET STRUCTURE
================================================================================

Training Dataset (for main.py)
-------------------------------

Prepare your dataset in this structure:

dataset/
├── action_challenge.csv
├── video1.mp4
├── video2.mp4
├── label1.txt
└── label2.txt

CSV Format (action_challenge.csv):
    video_file,label_file
    video1.mp4,label1.txt
    video2.mp4,label2.txt
    video3.mp4,

Label File Format (.txt):
    <class_id> <x1> <y1> <x2> <y2>
    
    class_id: Integer class ID (e.g., 0, 1, 2, ...)
    x1, y1, x2, y2: Bounding box coordinates in pixels (xyxy format)

Example (label1.txt):
    0 150 200 450 600

Note: Class IDs are automatically detected and mapped during training.
Configure your action classes in Config.CLASS_NAMES in main.py


Testing Dataset (for test.py)
------------------------------

For inference, you only need video files:

test_videos/
├── video1.mp4
├── video2.avi
└── video3.mov

Supported formats: .mp4, .avi, .mov, .mkv


USAGE
================================================================================

TRAINING (main.py)
------------------

Train the model on your custom action detection dataset:

python3 main.py --input_dir /path/to/dataset [--output_dir ./output]

Arguments:
    --input_dir    Dataset directory with videos, labels, CSV (required)
    --output_dir   Output directory (default: ./output)

Expected Files in input_dir:
    • action_challenge.csv (must be present)
    • Video files referenced in CSV
    • Label files referenced in CSV

Training Outputs (saved to output_dir):
    • split.json - Train/validation split information
    • best_model.pth - Best model checkpoint
    • training_results.csv - Training metrics per epoch
    • test_predictions.csv - Final validation predictions

Training Configuration:
    Epochs: 50 | Batch Size: 2 | Learning Rate: 5e-5
    Validation Split: 20% | Frames per Video: 16
    Image Size: 224×224 | Optimizer: AdamW with Cosine Annealing

Note: Modify Config class in main.py to customize hyperparameters


TESTING/INFERENCE (test.py)
----------------------------

Run inference on video files and generate predictions:

python3 test.py \
  --input_dir /path/to/videos \
  --output_file predictions.csv \
  [--model_path best_model.pth] \
  [--score_threshold 0.3]

Arguments:
    --input_dir        Directory containing video files (required)
    --output_file      Output CSV file path (required)
    --model_path       Model checkpoint path (default: best_model.pth)
    --score_threshold  Detection confidence threshold (default: 0.3)

Outputs:
    1. predictions.csv
       Detection results with class predictions and bounding boxes
       Format: video_name,pred_class_1,x1_1,y1_1,x2_1,y2_1,pred_class_2,...
    
    2. annotated_videos/
       Videos with bounding boxes and labels drawn on frames
    
    3. test_log_YYYYMMDD_HHMMSS.txt
       Detailed execution log with timestamps

Bounding Box Format:
    Coordinates are in normalized xyxy format (0-1 range)
    Multiply by video width/height to get pixel coordinates

Example Output (predictions.csv):
    video_name,pred_class_1,x1_1,y1_1,x2_1,y2_1
    video1.mp4,action_0,0.234,0.456,0.678,0.890
    video2.mp4,no_object,0.0,0.0,0.0,0.0
    video3.mp4,action_1,0.345,0.567,0.789,0.901


MODEL ARCHITECTURE
================================================================================

Architecture Components:

Backbone: R(2+1)D-18
    • 3D CNN with factorized convolutions
    • Pretrained on Kinetics-400 dataset
    • Extracts spatio-temporal features

Encoder:
    • Learned positional embeddings
    • 1×1 3D convolution for projection
    • Layer normalization + dropout

Decoder: Transformer Decoder
    • 4 layers with 8 attention heads
    • 2048-dimensional feedforward network
    • Cross-attention between queries and video features

Prediction Heads:
    • Classification Head: Action class probabilities
    • Bounding Box Head: Normalized (cx, cy, w, h) coordinates

Object Queries:
    • 12 learnable query embeddings
    • Each query can predict one action instance

Training Strategy:
    • Loss: DETR-style (Classification + 5×L1 + 2×GIoU)
    • Matching: Hungarian algorithm for bipartite matching
    • Optimizer: AdamW with weight decay
    • Scheduler: Cosine annealing learning rate
    • Precision: Mixed precision (FP16/FP32) with AMP

Model Statistics:
    Total Parameters: 40,509,220
    Model Size: ~155 MB
    Hidden Dimension: 256
    Number of Queries: 12


HOW IT WORKS
================================================================================

Training Pipeline:
    1. Load video clips and annotations
    2. Sample 16 frames uniformly from each video
    3. Extract features using R(2+1)D backbone
    4. Process through transformer decoder
    5. Match predictions to ground truth using Hungarian algorithm
    6. Compute combined loss (classification + localization)
    7. Update model weights via backpropagation

Inference Pipeline:
    1. Load video and sample 16 frames
    2. Forward pass through the model
    3. Apply confidence threshold to filter predictions
    4. Convert normalized boxes to pixel coordinates
    5. Draw bounding boxes on frames
    6. Save annotated video and CSV results


ADVANCED USAGE
================================================================================

Batch Processing Multiple Directories:

for video_dir in dir1/ dir2/ dir3/; do
    python3 test.py --input_dir $video_dir --output_file ${video_dir}_results.csv
done

Custom Confidence Threshold:

python3 test.py --input_dir ./videos --output_file out.csv --score_threshold 0.5

Force CPU Inference:

CUDA_VISIBLE_DEVICES="" python3 test.py --input_dir ./videos --output_file out.csv

Modify Action Classes:

Edit main.py Config class:
    NUM_CLASSES = 3  # Number of action classes
    CLASS_NAMES = ["action_A", "action_B", "action_C", "no_object"]


TROUBLESHOOTING
================================================================================

Out of Memory Error:
    Solution: Edit main.py Config class and set BATCH_SIZE = 1

CUDA Not Available:
    Info: Model automatically detects and uses CPU
    Note: Inference will be slower but functional (~5-10s per video)

Video Read Errors:
    • Install additional codecs: sudo apt-get install libavcodec-extra
    • Verify video integrity: ffmpeg -i video.mp4 -f null -
    • Check supported formats: .mp4, .avi, .mov, .mkv

Label Format Errors:
    • Ensure exactly 5 space-separated values per line
    • Verify class IDs are integers
    • Check bounding box coordinates are within video dimensions

Model Loading Errors:
    • Verify best_model.pth exists in specified path
    • Check file size (~155 MB)
    • Re-download from Hugging Face if corrupted


TECHNICAL DETAILS
================================================================================

Based on State-of-the-Art Research:
    • DETR (Detection Transformer) - Carion et al., ECCV 2020
    • STAR (Spatio-Temporal Action Recognition) - Gritsenko et al., CVPR 2024
    • R(2+1)D Networks - Tran et al., CVPR 2018

Architecture Adaptations:
    ✓ Simplified for practical deployment
    ✓ Clip-level prediction (16 frames per video)
    ✓ Single bounding box per action instance
    ✓ Optimized for small to medium datasets
    ✓ CPU-compatible for edge deployment

Key Design Choices:
    • Hungarian matching ensures one-to-one correspondence
    • GIoU loss improves localization quality
    • Mixed precision training speeds up convergence
    • Balanced batch sampling prevents class imbalance


CUSTOMIZATION
================================================================================

To adapt this model for your specific use case:

1. Modify Action Classes:
   Edit Config.CLASS_NAMES in main.py:
       CLASS_NAMES = ["your_action_1", "your_action_2", "no_object"]

2. Adjust Model Capacity:
   Edit Config in main.py:
       NUM_QUERIES = 20  # More queries for multiple instances
       HIDDEN_DIM = 512  # Larger model capacity

3. Change Training Duration:
   Edit Config in main.py:
       NUM_EPOCHS = 100  # Longer training
       LEARNING_RATE = 1e-4  # Different learning rate

4. Modify Evaluation Threshold:
   Pass argument to test.py:
       --score_threshold 0.7  # Higher confidence threshold


CITATION
================================================================================

If you use this code in your research or project, please cite:

@software{video_action_transformer_2026,
  author = {Ayush Saun and Aman Sharma},
  title = {STAR: Spatio-Temporal Action Recognition},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/video-action-transformer}
}

Original DETR paper:

@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and 
          Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={ECCV},
  year={2020}
}


LICENSE
================================================================================

MIT License - See LICENSE file for details

This implementation is for educational and research purposes.


CONTRIBUTING
================================================================================

Contributions are welcome! Areas for improvement:
    • Multi-instance detection (multiple actions per video)
    • Temporal localization (per-frame predictions)
    • Additional action categories
    • Data augmentation strategies
    • Model compression (quantization, pruning)
    • Real-time inference optimization

Please open an issue or submit a pull request.


ACKNOWLEDGMENTS
================================================================================

• Meta AI Research for DETR architecture
• Google Research for STAR and R(2+1)D contributions
• PyTorch team for deep learning framework
• Hugging Face for model hosting platform
• Open-source computer vision community


SUPPORT
================================================================================

Issues: https://github.com/YOUR_USERNAME/video-action-transformer/issues
Model: https://huggingface.co/YOUR_USERNAME/video-action-transformer

For questions or bug reports:
    • Check log files for detailed error messages
    • Verify dataset format matches documentation
    • Ensure model weights are correctly downloaded
    • Review troubleshooting section above


================================================================================
                        End of Documentation
================================================================================
