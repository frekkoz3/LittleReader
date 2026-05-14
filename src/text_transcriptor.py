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
import easyocr

def run_ocr_on_patches(
    patches: list[dict],
    reader: easyocr.Reader,
):
    """
    Run OCR on extracted patches.

    Parameters
    ----------
    patches : list[int]
        Output of cut_patches().

    reader : easyocr.Reader
        Initialized EasyOCR reader.

    Returns
    -------
    list[dict]
        OCR results per patch.
    """
    ocr_results = []

    for patch in patches:

        text_result = reader.readtext(
            patch["crop"],
            detail=1,
            paragraph=True,
        )

        ocr_results.append(
            {
                "class_id": patch["class_id"],
                "bbox_xyxy": patch["bbox_xyxy"],
                "ocr": text_result,
            }
        )

    return ocr_results

if __name__ == '__main__':
    pass