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
from dotenv import load_dotenv

load_dotenv()

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def run_ocr_on_patches(
    patches: list[dict],
    reader,
):
    """
    Run OCR on extracted patches.

    Parameters
    ----------
    patches : list[int]
        Output of cut_patches().

    reader : easyocr.Reader
        Initialized doctr reader.

    Returns
    -------
    list[dict]
        OCR results per patch.
    """
    ocr_results = []

    for i, patch in enumerate(patches):
        
        doc = DocumentFile.from_images(f"crops/tile{i}.png")
        text_result = reader(doc).render()

        ocr_results.append(
            {
                "class_id": patch["class_id"],
                "bbox_xyxy": patch["bbox_xyxy"],
                "ocr": text_result,
            }
        )

    return ocr_results

from matplotlib import pyplot as plt

if __name__ == '__main__':
    img_path = "imgs/tile_proof.jpg"
    # Load the document image
    doc = DocumentFile.from_images(img_path)

    predictor = ocr_predictor(pretrained=True)

    result = predictor(doc)

    result.show()

    string_result = result.render()
    print(string_result)