import re

def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace and unwanted characters.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The cleaned text.
    """
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into chunks with overlapping context.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of characters per chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
    
    Returns:
        list: A list of text chunks.
    """
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

if __name__ == "__main__":
    # For testing, use a sample text from the sample PDF file if needed,
    # here we simulate with repeated sample content.
    sample_text = "This is a sample text from the PDF file. " * 50
    chunks = chunk_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...\n")