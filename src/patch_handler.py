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
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import math

def cut_patches(
    img: np.ndarray,
    layout: Iterable[tuple[int, tuple[float, float, float, float]]],
    save_images: bool = False,
    o_path: str | Path | None = None,
) -> list[dict]:
    """
    Extract image patches from YOLO-layout annotations.

    The function parses YOLO-style normalized annotations and extracts
    crops corresponding only to:
        - class 2 -> Section Title
        - class 3 -> Column

    YOLO annotation format:
        (class_id, (x_center, y_center, width, height))

    where all coordinates are normalized in the range [0, 1]
    with respect to the full image dimensions.

    Supported classes:
        0 -> Header
        1 -> Section
        2 -> Section Title
        3 -> Column
        4 -> Banner
        5 -> Footer

    Parameters
    ----------
    img : np.ndarray
        Input image in OpenCV format (H, W, C).

    layout : Iterable[tuple[int, tuple[float, float, float, float]]]
        Iterable containing YOLO annotations.
        Example:
            [
                (2, (0.5, 0.2, 0.4, 0.1)),
                (3, (0.5, 0.6, 0.8, 0.3)),
            ]

    save_images : bool, optional
        If True, extracted crops are saved to disk.
        Default is False.

    o_path : str | Path | None, optional
        Output directory where crops will be saved.
        If None and `save_images=True`, the directory "crops/"
        will be created and used.

    Returns
    -------
    list[dict]
        A list containing one dictionary per extracted patch.

        Each dictionary contains:
            {
                "class_id": int,
                "bbox_xyxy": (x1, y1, x2, y2),
                "crop": np.ndarray,
                "path": str | None
            }

    Notes
    -----
    - Bounding boxes are automatically clamped to image boundaries.
    - Invalid or empty crops are skipped.
    - For now we are only interested in class 2 and class 3 (Section Titles and Columns)
    """

    TARGET_CLASSES = {2, 3}

    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    h, w = img.shape[:2]

    output_dir = Path(o_path) if o_path is not None else Path("crops")

    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    patches = []

    for idx, annotation in enumerate(layout):

        try:
            cls, xywhn = annotation
            x_center, y_center, bw, bh = xywhn
        except Exception as e:
            raise ValueError(
                f"Invalid annotation format at index {idx}: {annotation}"
            ) from e

        cls = int(cls)

        # Ignore unwanted classes
        if cls not in TARGET_CLASSES:
            continue

        # Convert normalized YOLO coordinates -> pixel coordinates
        x_center_px = x_center * w
        y_center_px = y_center * h
        bw_px = bw * w
        bh_px = bh * h

        x1 = int(round(x_center_px - bw_px / 2))
        y1 = int(round(y_center_px - bh_px / 2))
        x2 = int(round(x_center_px + bw_px / 2))
        y2 = int(round(y_center_px + bh_px / 2))

        # Clamp coordinates to image boundaries
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Extract crop
        crop = img[y1:y2, x1:x2]

        # Skip empty crops
        if crop.size == 0:
            continue

        saved_path = None

        if save_images:
            filename = f"crop_{idx}_class_{cls}.jpg"
            saved_path = str(output_dir / filename)

            success = cv2.imwrite(saved_path, crop)

            if not success:
                raise IOError(f"Failed to save crop to: {saved_path}")

        patches.append(
            {
                "class_id": cls,
                "bbox_xyxy": (x1, y1, x2, y2),
                "crop": crop,
                "path": saved_path,
            }
        )

    # order patches first by y then by x
    patches.sort(key=lambda p: (p["bbox_xyxy"][1], p["bbox_xyxy"][0]))

    return patches

def compute_iou(
    box_a : tuple[int, int, int, int],
    box_b :tuple[int, int, int, int]
) -> int:
    """
    Compute IoU (intersection over union) between two boxes in xyxy format.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    # Areas
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0

    return inter_area / union

def remove_overlapping_patches(
    patches : list[dict],
    iou_threshold : int = 0.2
) -> list[dict]:
    """
    Remove overlapping pateches from the given list of patches.
    The overlap is computed by comparing the IOU (Intersetion over Union) with a given threshold.

    Parameters
    ----------
    patches : list[dict]
        A list containing one dictionary per patch.

        Each dictionary contains:
            {
                "class_id": int,
                "bbox_xyxy": (x1, y1, x2, y2),
                "crop": np.ndarray,
                "path": str | None
            }

    Returns
    -------
    list[dict]
        A list containing one dictionary per non-overlapping patch.

        Each dictionary contains:
            {
                "class_id": int,
                "bbox_xyxy": (x1, y1, x2, y2),
                "crop": np.ndarray,
                "path": str | None
            }
    """
    # Remove highly-overlapping boxes
    filtered_patches = []

    for patch in patches:

        keep = True

        for kept_patch in filtered_patches:

            iou = compute_iou(
                patch["bbox_xyxy"],
                kept_patch["bbox_xyxy"],
            )

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            filtered_patches.append(patch)

    patches = filtered_patches #we must transform this in "almost square patches"

    return patches

def cut_almost_squared_tiles(
    patches: list[dict]
) -> list[dict]:
    """
    Cut patches into near-square tiles using a simple heuristic based on the percentge of black pixels in the line.
    
    Parameters 
    ---------- 
    patches: list[dict] 
        A list containing one dictionary per patch.
        Each dictionary contains: 
        { 
            "class_id": int, 
            "bbox_xyxy": (x1, y1, x2, y2), 
            "crop": np.ndarray, "path": str | None 
        }
    
    Returns 
    ------- 
    list[dict] 
        A list containing one dictionary per extracted tiles.
        Each dictionary contains:
        {
            "class_id": int,
            "bbox_xyxy": (x1, y1, x2, y2),
            "crop": np.ndarray, "path": str | None 
        }
    """
    
    almost_squared_tiles = []

    for patch in patches:
        x1, y1, x2, y2 = patch["bbox_xyxy"]
        crop = patch["crop"]

        h, w = crop.shape[:2]

        # If already roughly square, keep as is
        if h <= w:
            almost_squared_tiles.append(patch)
            continue

        step = w  # target square height
        start = 0

        while start < h:
            end = min(start + step, h)

            # Try to adjust cut to avoid splitting text
            while end < h:
                row = crop[end-1:end, :]
                black_pixels = np.sum(row == 0)
                ratio = black_pixels / (w)

                if ratio < 0.03:  # safe cut
                    break
                end += 1

            tile_crop = crop[start:end, :]

            tile = {
                "class_id": patch["class_id"],
                "bbox_xyxy": (x1, y1 + start, x2, y1 + end),
                "crop": tile_crop,
                "path": patch["path"],
            }

            almost_squared_tiles.append(tile)
            start = end  # move forward

    return almost_squared_tiles

if __name__ == '__main__':
    ...