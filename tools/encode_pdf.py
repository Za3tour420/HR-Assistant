import fitz
import base64

def encode_pdf(file_path: str) -> str:
    b64_pdf = None
    file_content = ""

    # Open PDF and read data
    doc = fitz.open(file_path)
    for page in doc:
        content = page.get_text()
        if content:
            file_content += content + "\n"
        
    # Encode if content
    if file_content:
        b64_pdf = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")
    return b64_pdf


def encode_pdf_stream(file_stream):
    b64_pdf = None
    file_content = ""

    # Open PDF from stream and read data
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    for page in doc:
        content = page.get_text()
        if content:
            file_content += content + "\n"
        
    # Encode if content
    if file_content:
        b64_pdf = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")
    return b64_pdf