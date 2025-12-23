import fitz
import base64

def encode_pdf(file_path: str, dpi: int = 200) -> list[str]:
    b64_page_images = []

    # Open PDF and treat pages as images
    with fitz.open(file_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            b64_page_images.append(
                base64.b64encode(img_bytes).decode("utf-8")
            )

    return b64_page_images
            


def encode_pdf_stream(file_stream, dpi: int = 200) -> list[str]:
    b64_page_images = []

    # Open PDF from stream
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            b64_page_images.append(
                base64.b64encode(img_bytes).decode("utf-8")
            )
    
    return b64_page_images