import PyPDF2

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a digital PDF using PyPDF2.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text

if __name__ == "__main__":
    # For testing, point to the sample file in govrag/samples/weinberg1.pdf
    import os
    sample_path = os.path.join(os.path.expanduser("~/Documents/govrag/samples"), "weinberg1.pdf")
    extracted_text = extract_text_from_pdf(sample_path)
    print("Extracted Text Preview:")
    print(extracted_text[:500])