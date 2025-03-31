import os
from preprocessing.text_extraction import extract_text_from_pdf
from preprocessing.ocr import perform_ocr_on_pdf

def ingest_pdf(file_path: str) -> str:
    """
    Ingests a PDF file and returns its extracted text.
    It first attempts digital text extraction; if that fails,
    it falls back to OCR extraction.
    """
    text = extract_text_from_pdf(file_path)
    if not text or len(text.strip()) == 0:
        print(f"Digital extraction failed for {file_path}. Falling back to OCR...")
        text = perform_ocr_on_pdf(file_path)
    return text

def ingest_sample_pdf():
    """
    Ingests the sample PDF file located at govrag/samples/weinberg1.pdf.
    """
    base_path = os.path.expanduser("~/Documents/govrag/samples")
    sample_file = os.path.join(base_path, "weinberg1.pdf")
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        return None
    return ingest_pdf(sample_file)

if __name__ == "__main__":
    extracted_text = ingest_sample_pdf()
    if extracted_text:
        print("Extracted Text Preview:")
        print(extracted_text[:500])