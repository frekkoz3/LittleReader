# OCR module

This module implements the core components of the Optical Character Recognition (OCR) pipeline used in the project.

## Pipeline

The OCR process follows a structured sequence of steps designed to extract and reconstruct textual information from document pages:

1. Page Input – The original document page is provided as input to the pipeline.
2. Layout Detection – The structural layout of the page is analyzed to identify distinct text regions.
3. Patch Creation – The detected regions are segmented into smaller image patches corresponding to individual textual elements.
4. Patch Ordering – The patches are arranged according to their logical reading order within the document.
5. Text Type Detection – Each patch is classified based on its semantic role (e.g., title, subtitle, body text).
6. Text Transcription – The textual content of each patch is transcribed using an OCR model.

The output of the pipeline is the reconstructed textual content of the document, enriched with structural and semantic information.

## Techniques

## Method Reference

The following tables summarize the methods provided by this module, including their parameters, return values, and descriptions.

### Layout detector

| Method Name | Parameters | Output | Description |
|-------------|------------|--------|-------------|
| `page_detector`   | `-`   | `-`  | Find the exact page to analyze, removing border's noise. |

### Patch handler

| Method Name | Parameters | Output | Description |
|-------------|------------|--------|-------------|
| `methodA`   | `x: int`   | `int`  | Computes the value of ... |

### Type classificator

| Method Name | Parameters | Output | Description |
|-------------|------------|--------|-------------|
| `methodA`   | `x: int`   | `int`  | Computes the value of ... |

### Text Transcriptor

| Method Name | Parameters | Output | Description |
|-------------|------------|--------|-------------|
| `methodA`   | `x: int`   | `int`  | Computes the value of ... |

## Tesseract Dependencies

In order to use Tesseract you need to download it.

- **Windows**

[UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

- **Linux**

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ita poppler-utils
```
