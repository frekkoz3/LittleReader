r"""
     _____ _               _         _____        _                 _   
    |  __ (_)             | |       |  __ \      | |               | |  
    | |__) |  ___ ___ ___ | | ___   | |  | | __ _| |_ __ _ ___  ___| |_ 
    |  ___/ |/ __/ __/ _ \| |/ _ \  | |  | |/ _` | __/ _` / __|/ _ \ __|
    | |   | | (_| (_| (_) | | (_) | | |__| | (_| | || (_| \__ \  __/ |_ 
    |_|   |_|\___\___\___/|_|\___/  |_____/ \__,_|\__\__,_|___/\___|\__|
                                                                                                                        
    University of Trieste

    Research code for the Piccolo Dataset initiative, aimed at supporting
    data-driven research through the creation and curation of structured
    datasets for analysis and machine learning experimentation.
"""

from __future__ import annotations

import cv2
import easyocr
import torch
from ultralytics import YOLO, RTDETR
import numpy as np
from paddleocr import PaddleOCR

from src.ocr.patch_handler import cut_patches, remove_overlapping_patches, cut_almost_squared_tiles
from src.ocr.text_transcriptor import run_ocr_on_patches

def layout_detector(
    model,
    image: np.ndarray,
    conf_threshold: float = 0.25,
) -> list:
    """
    Run layout detection on an image.

    Parameters
    ----------
    model :
        Ultralytics YOLO or RTDETR model.

    image : npndarray
        Image of the newspaper

    conf_threshold : float
        Minimum confidence required to keep detections.

    Returns
    -------
    list
        Layout annotations.

        A layout annotation is a list that follows YOLO format:
            (class_id, (x_center, y_center, width, height))
    """

    if image is None:
        print(f"[WARNING] Failed to read: {img_path}")
        return

    results = model(image)

    result = results[0]

    xywhn = result.boxes.xywhn.cpu()
    classes = result.boxes.cls.cpu()
    confs = result.boxes.conf.cpu()

    layout = []

    for cls, box, conf in zip(classes, xywhn, confs):

        conf = float(conf)

        if conf < conf_threshold:
            continue

        cls = int(cls.item())

        box = tuple(float(v) for v in box.tolist())

        layout.append((cls, box))

    return layout
    
if __name__ == "__main__":

    model_name = "1024_1_yolo26"

    img_path = "imgs/00001.png"

    image = cv2.imread(str(img_path))

    # Load model
    model = YOLO(f"models/{model_name}.pt")

    #model = RTDETR(f"models/{model_name}.pt")

    layout = layout_detector(model, image)

    # Extract patches
    patches = cut_patches(
        image,
        layout,
        save_images=True
    )

    non_overlapping_patches = remove_overlapping_patches(patches)

    tiles = cut_almost_squared_tiles(non_overlapping_patches)

    for i, tile in enumerate(tiles):
        cv2.imwrite(f"crops/tile{i}.png", tile["crop"])

    reader = easyocr.Reader(["it"])

    ocr_results = run_ocr_on_patches(
        tiles,
        reader
    )

    # Print OCR
    for i, result in enumerate(ocr_results):

        print("\n" + "=" * 60)
        print(f"TILE {i}")
        print("=" * 60)

        for detection in result["ocr"]:

            try:
                _, text, confidence = detection

                print(
                    f"[{confidence:.2f}] {text}"
                )

            except Exception:
                print(detection)