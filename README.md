# EpiSAM: Character Segmentation in Challenging Stone Inscriptions

<p align="center">
  <a href="https://arxiv.org/pdf/2606.28859">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=adobeacrobatreader&logoColor=white"></a>
  <a href="https://ihdia.iiit.ac.in/episam/">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green"></a>
  <a href="https://huggingface.co/datasets/keyfinder08/EpiSAM_Inscriptions_Dataset">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-blue?style=flat&logo=huggingface&logoColor=yellow"></a>
  <a href="https://ihdia.iiit.ac.in/episam/dataset.html">
    <img src="https://img.shields.io/badge/Dataset-Website-green?style=flat&logo=Google%20chrome&logoColor=white"></a>
  <a href="https://huggingface.co/keyfinder08/EpiSAM_Inscriptions_Project"> <img src="https://img.shields.io/badge/Weights-HuggingFace-yellow?style=flat&logo=huggingface&logoColor=black"> </a>
  <img src="https://komarev.com/ghpvc/?username=keyfinder08-episam&label=VISITORS&color=0e75b6&style=flat" alt="Visitor Count">
</p>


<h3 align="center"><b>Accepted at ICDAR 2026 (CORE A)</b></h3>

This repository contains the codebase for dataset preprocessing, training, and evaluation of **EpiSAM** (Segment Anything Model fine-tuned for historical epigraphy and character segmentation).

---

## 📋 Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Dataset Directory Structure](#2-dataset-directory-structure)
3. [Dataset Preprocessing](#3-dataset-preprocessing)
4. [Training the Model](#4-training-the-model)
5. [Evaluating the Model](#5-evaluating-the-model)
6. [Line Grouping using 2-Hop Graph Classification](#6-line-grouping-using-2-hop-graph-classification)
7. [Citation](#7-citation)
8. [Acknowledgement](#8-acknowledgement)

---

## 1. Environment Setup

You can set up the required Conda environment using the provided `environment.yml` file.

```bash
# 1. Create the conda environment from the environment file
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate episam
```

---

## 2. Dataset Directory Structure

Ensure your dataset directory structure follows this format:

```text
episam_dataset/
├── whole_images/
│   ├── train/            # Whole input images for training split
│   └── val/              # Whole input images for validation split
├── binary_images/        # Preprocessed binarized images (optional, searched recursively)
├── chars/
│   ├── train/            # Character polygons & metadata per inscription (train split)
│   └── val/              # Character polygons & metadata per inscription (val split)
└── associated_centroids_updated/
    ├── train/            # Associated centroids JSON files (train split)
    └── val/              # Associated centroids JSON files (val split)
```

---

## 3. Dataset Preprocessing

Before training or evaluation, process the dataset to generate `.npz` mask arrays and master JSON index files. You must run the `prepare_dataset.py` script **twice**—once for the `train` split and once for the `val` split.

### Step 3.1: Process Train Split

```bash
python dataset_scripts/prepare_dataset.py \
  --associations-dir path/to/episam_dataset/associated_centroids_updated \
  --convex-hulls-dir path/to/episam_dataset/chars \
  --images-dir path/to/episam_dataset/whole_images \
  --binarized-dir path/to/episam_dataset/binary_images \
  --output-char-masks processed_data/char_masks_train \
  --output-line-masks processed_data/line_masks_train \
  --output-prompt-masks processed_data/prompt_masks_train \
  --output-neighbor-masks processed_data/neighbor_masks_train \
  --output-index processed_data/master_index_train.json \
  --splits train
```

### Step 3.2: Process Validation Split

```bash
python dataset_scripts/prepare_dataset.py \
  --associations-dir path/to/episam_dataset/associated_centroids_updated \
  --convex-hulls-dir path/to/episam_dataset/chars \
  --images-dir path/to/episam_dataset/whole_images \
  --binarized-dir path/to/episam_dataset/binary_images \
  --output-char-masks processed_data/char_masks_val \
  --output-line-masks processed_data/line_masks_val \
  --output-prompt-masks processed_data/prompt_masks_val \
  --output-neighbor-masks processed_data/neighbor_masks_val \
  --output-index processed_data/master_index_val.json \
  --splits val
```

---

## 4. Training the Model

Train the model using `train.py`. Pass the pretrained SAM weights checkpoint (`--checkpoint`) and the generated training & validation index JSON files.

```bash
python train.py \
  --model_type vit_b \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --train_index_file processed_data/master_index_train.json \
  --val_index_file processed_data/master_index_val.json \
  --checkpoint_dir ./checkpoints/run_01 \
  --comparison_images_dir ./comparison_outputs/run_01 \
  --epochs 100 \
  --lr 1e-5 \
  --batch_size 1 \
  --device cuda:0
```

### Key Training Flags:
* `--checkpoint`: Path to pre-trained SAM checkpoint (e.g. `sam_vit_b_01ec64.pth`).
* `--train_index_file` / `-t`: Path to master index JSON for training (`master_index_train.json`).
* `--val_index_file` / `-v`: Path to master index JSON for validation (`master_index_val.json`).
* `--use_wandb` / `-w`: Optional flag to enable Weights & Biases logging.
* `--resume_from`: Path to a checkpoint `.pth` file if resuming previous training runs.

---

## 5. Evaluating the Model

To evaluate a trained checkpoint on your validation set and save visualizations:

```bash
python evaluate.py \
  --model_type vit_b \
  --checkpoint ./checkpoints/run_01/best_model.pth \
  --index_file processed_data/master_index_val.json \
  --output_dir ./evaluation_results/ \
  --device cuda:0 \
  --pred_iou_threshold 0.3
```

---

## 6. Line Segmentation using 2-Hop Graph Classification
Note: This is not a part of the EpiSAM paper, but is something we used in the demo application. The demo was accepted as part of the Demo track at the DAS workshop conducted within ICDAR '26. Refer to [EpiSAM_DAS.pdf](EpiSAM_DAS.pdf) for more details.
<img width="1515" height="414" alt="image" src="https://github.com/user-attachments/assets/80309d6a-ed97-4fdd-9f62-667320bbda5d" />


The `2_hop_graphs_lines.py` script constructs a 2-hop graph over character predictions and performs edge classification using a heuristic similarity threshold with common-neighbour veto to group characters into text lines.

### Usage Example

```bash
python 2_hop_graphs_lines.py \
  --precomputed_dir ./evaluation_results/ \
  --image_root path/to/episam_dataset/whole_images/val \
  --index_file processed_data/master_index_val.json \
  --output_dir ./graph_lines_output/ \
  --k 5 \
  --similarity_method dice \
  --threshold 0.4
```

### Key Arguments:
* `--precomputed_dir`: Path containing precomputed predictions (`.npz` files output from evaluation).
* `--image_root`: Path to original images corresponding to the evaluated set.
* `--index_file`: Master index JSON file with line annotations.
* `--k`: Number of 1-hop nearest neighbors to construct the base graph (default: `5`).
* `--similarity_method`: Mask overlap similarity metric (`dice`, `iou`, or `ios`).
* `--threshold`: Threshold $t$ for edge classification (default: `0.4`).


## 7. Citation

If you use our dataset in your research, please cite the following papers:

```bibtex
@InProceedings{10.1007/978-3-032-36039-7_18,
author="Sharma, Arnav
and Jena, Pratyush
and Joseph, Amal
and Sarvadevabhatla, Ravi Kiran",
title="EpiSAM: Character Segmentation in Challenging Stone Inscriptions",
booktitle="Document Analysis and Recognition -- ICDAR 2026",
year="2027",
publisher="Springer Nature Switzerland",
address="Cham",
pages="299--315",
isbn="978-3-032-36039-7"
}


@inproceedings{Jena_2025,
series={ICVGIP 2025},
title={Unveiling Text in Challenging Stone Inscriptions: A Character-Context-Aware Patching Strategy for Binarization},
url={http://dx.doi.org/10.1145/3774521.3774539},
DOI={10.1145/3774521.3774539},
booktitle={Proceedings of the Sixteen Indian Conference on Computer Vision, Graphics and Image Processing},
publisher={ACM},
author={Jena, Pratyush and Joseph, Amal and Sharma, Arnav and Sarvadevabhatla, Ravi Kiran},
year={2025},
month=Dec,
pages={1–9},
collection={ICVGIP 2025}
}
```

---

## 8. Acknowledgement

We sincerely acknowledge **The Mythic Society Bengaluru** for providing the inscription images used in this work. These resources are part of the *Inscriptions 3D Digital Conservation Project*, an initiative aimed at preserving and digitizing valuable epigraphic heritage.

For more information about the project, please visit:

[Akshara Bhandara – Inscriptions 3D Digital Conservation Project](https://mythicsociety.github.io/AksharaBhandara/).

---
