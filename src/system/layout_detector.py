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
from doctr.models import ocr_predictor
import torch
from ultralytics import YOLO, RTDETR
import numpy as np

from huggingface_hub import hf_hub_download

from src.system.patch_handler import cut_patches, remove_overlapping_patches, cut_almost_squared_tiles
from src.system.text_transcriptor import run_ocr_on_patches

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

    using_yolo = False

    repo_id = "frekko/paper_model_yolo26" if using_yolo else "frekko/paper_model_rt_detr"
    model_name = "1024_0_v2_yolo26.pt" if using_yolo else "1024_0_v2_rt_detr.pt"

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_name,
        token=True
    )

    model = YOLO(model_path) if using_yolo else RTDETR(model_path)

    img_path = "imgs/proof.png"

    image = cv2.imread(str(img_path))

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
        cv2.imwrite(f"crops/tile_{i}.png", tile["crop"])

    reader = ocr_predictor(pretrained=True)

    ocr_results = run_ocr_on_patches(
        tiles,
        reader
    )

    final_txt = ""

    for i, result in enumerate(ocr_results):

        for detection in result["ocr"]:

            try:
                text = detection

                final_txt+= text

            except Exception:
                print(detection)

    final_txt = final_txt.replace("\n", " ")

    final_txt = final_txt.strip()

    print(final_txt)
        
    with open("text_proof.txt", "w+") as f:
        f.write(final_txt)
    