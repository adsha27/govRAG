import os
from preprocessing.text_extraction import extract_text_from_pdf
from preprocessing.ocr import perform_ocr_on_pdf


def ingest_pdf(file_path: str, return_pages: bool = False):
    """
    Ingest a PDF. Returns text string, or (text, pages) if return_pages=True.
    pages is a list of (page_num, page_text) tuples.
    Falls back to OCR if digital extraction yields nothing.
    """
    pages = _extract_pages(file_path)

    if not pages:
        print(f"Digital extraction failed for {file_path}. Falling back to OCR...")
        full_text = perform_ocr_on_pdf(file_path)
        if return_pages:
            return full_text, [(0, full_text)]
        return full_text

    full_text = " ".join(text for _, text in pages)

    if return_pages:
        return full_text, pages
    return full_text


def _extract_pages(file_path: str) -> list[tuple[int, str]]:
    """Extract per-page text using PyMuPDF if available, else fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
        doc.close()
        return pages
    except ImportError:
        pass

    # Fallback: extract all text without page tracking
    text = extract_text_from_pdf(file_path)
    if text and text.strip():
        return [(1, text)]
    return []
