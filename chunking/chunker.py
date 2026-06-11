import re


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Original character-based chunker. Kept for compatibility."""
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def chunk_text_sentences(
    text: str,
    pages: list[tuple[int, str]] | None = None,
    target_size: int = 400,
    overlap_sentences: int = 2,
) -> tuple[list[str], list[int]]:
    """
    Sentence-aware chunker. Splits on sentence boundaries so chunks are
    semantically complete. Returns (chunks, page_numbers).

    If pages is a list of (page_num, page_text), tracks which page each
    chunk came from. Otherwise all chunks get page 0.
    """
    if pages is not None:
        all_chunks = []
        all_pages = []
        for page_num, page_text in pages:
            c = _chunk_by_sentences(page_text, target_size, overlap_sentences)
            all_chunks.extend(c)
            all_pages.extend([page_num] * len(c))
        return all_chunks, all_pages

    chunks = _chunk_by_sentences(text, target_size, overlap_sentences)
    return chunks, [0] * len(chunks)


def _chunk_by_sentences(text: str, target_size: int, overlap: int) -> list[str]:
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > target_size and current:
            chunks.append(' '.join(current))
            current = current[-overlap:] if overlap > 0 else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(' '.join(current))

    return [c for c in chunks if c.strip()]
