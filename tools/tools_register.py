from langchain_core.tools import tool
from tools.encode_pdf import encode_pdf, encode_pdf_stream

@tool("encode_pdf_tool")
def encode_pdf_tool(file_path: str) -> list[str]:
    """Encodes a PDF file located at the given file path into a list of base64 strings of its page images."""
    return encode_pdf(file_path)