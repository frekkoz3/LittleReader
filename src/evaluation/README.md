# Evaluation Module

This directory contains the evaluation scripts for the **LittleReader System**, including both **benchmark comparisons against external systems** and **ablation studies** to assess individual components of the pipeline.

---

## Overview

The evaluation is organized into two main parts:

- **Benchmarks**: comparison against state-of-the-art OCR and document understanding systems  
- **Ablation Studies**: evaluation of individual modules within the LittleReader pipeline  

---

## Metrics

### Document Parsing Module

We evaluate layout detection performance using standard object detection metrics:

- mAP@0.5  
- mAP@0.7  
- mAP@0.9  
- mAP@[0.5:0.95] (step 0.05)  

These metrics measure the quality of detected layout regions across different IoU thresholds.

---

### Document Reconstruction System

We evaluate text reconstruction quality using standard OCR metrics:

- CER (Character Error Rate)  
- WER (Word Error Rate)  

These metrics quantify transcription accuracy compared to ground truth text.

---

### Computational Efficiency

We report system efficiency across three dimensions:

- Model Complexity (e.g., number of parameters / architectural complexity)
- Training Cost (compute and time requirements)
- Inference Cost (latency and runtime resource usage)

---

## Benchmarks

We evaluate LittleReader against both classical OCR systems and modern document understanding pipelines.

---

### Datasets

The following datasets are used for benchmarking:

1. DocLayNet  
2. PRImA Document Layout Analysis Dataset  
3. PubLayNet  
4. Italian Document Dataset (to be specified)  

---

### Baseline Systems

We compare against the following systems:

- Tesseract OCR  
- PaddleOCR (with and without VLM enhancements)  
- EasyOCR  

---

### Dataset-Specific Setup

#### DocLayNet

We apply a class remapping to align dataset labels with the LittleReader taxonomy.  
*(Details to be specified here.)*

---

#### PRImA Dataset

We perform a custom class mapping from PAGE XML region types to the unified label space used in our evaluation framework.  
*(Details to be specified here.)*

---

#### PubLayNet

We map PubLayNet’s native classes to the LittleReader label schema to ensure consistent evaluation across datasets.  
*(Details to be specified here.)*

---

## Ablation Studies

The LittleReader system is composed of several modular components. We evaluate their contribution by selectively disabling parts of the pipeline.

---

### Pipeline Components

1. Document Layout Parsing (using finetuned rt-detr)
2. Patch Handling  
   - crop filtering  
   - near-square tiling strategy  
3. Text Extraction (using Doctr)  
4. Text Post-processing  
   - VLM-based correction  
   - partial text classification  

---

### Ablation Objective

The ablation studies measure performance degradation when removing or isolating components of the pipeline, in order to quantify:

- contribution of layout parsing  
- impact of patch-based processing  
- effectiveness of OCR module  
- value added by VLM-based correction  

---

## Notes

- All evaluations are performed under consistent preprocessing and annotation normalization.  
- Metrics are computed separately for each dataset and then aggregated where applicable.  
- Class mappings are explicitly defined per dataset to ensure reproducibility.
- The evaluation is done on a computational node provided of two A100.
