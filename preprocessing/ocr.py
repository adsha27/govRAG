import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

def perform_ocr_on_pdf(file_path: str, dpi: int = 300) -> str:
    """
    Performs OCR on a scanned PDF using PaddleOCR.
    Converts each page of the PDF to an image and extracts text.
    
    Args:
        file_path (str): The path to the PDF file.
        dpi (int): Dots per inch for image conversion (default is 300).
    
    Returns:
        str: The combined OCR-extracted text from all pages.
    """
    # Initialize PaddleOCR for English. Adjust lang if needed.
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    
    try:
        pages = convert_from_path(file_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return ""
    
    all_text = []
    
    for idx, page in enumerate(pages):
        print(f"Processing page {idx+1}/{len(pages)}...")
        try:
            result = ocr_model.ocr(page, rec=True)
            page_text = " ".join([line[1][0] for line in result])
            all_text.append(page_text)
        except Exception as e:
            print(f"Error processing page {idx+1}: {e}")
    
    return "\n\n".join(all_text)

if __name__ == "__main__":
    # For testing, use the sample file in govrag/samples/weinberg1.pdf
    sample_file = os.path.join(os.path.expanduser("~/Documents/govrag/samples"), "weinberg1.pdf")
    ocr_text = perform_ocr_on_pdf(sample_file)
    print("OCR Extracted Text Preview:")
    print(ocr_text[:500])