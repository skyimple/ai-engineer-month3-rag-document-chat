import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF


class PDFProcessor:
    """Process PDF documents and extract text with metadata"""

    def __init__(self):
        self.supported_extensions = {".pdf"}

    def is_supported(self, file_path: str) -> bool:
        """Check if file is a supported PDF"""
        return Path(file_path).suffix.lower() in self.supported_extensions

    def extract_text_from_page(self, page: fitz.Page) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a single page.

        Returns:
            Tuple of (text_content, metadata_dict)
        """
        # Get page text
        text = page.get_text("text")

        # Get page metadata
        page_label = page.get_label() or str(page.number + 1)

        metadata = {
            "page_label": page_label,
            "page_number": page.number + 1,
            "text_length": len(text),
        }

        return text, metadata

    def extract_images_from_page(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page (without OCR).

        Returns list of image dicts with image_index, width, height, bbox.
        """
        images = []
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)

            images.append({
                "image_index": image_index,
                "xref": xref,
                "width": base_image.get("width", 0),
                "height": base_image.get("height", 0),
                "colorspace": base_image.get("colorspace", ""),
                "bpc": base_image.get("bpc", 0),
                "ext": base_image.get("ext", ""),
            })

        return images

    def process_pdf(
        self,
        file_path: str,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF file and extract all content.

        Args:
            file_path: Path to the PDF file
            file_name: Optional override for file name

        Returns:
            Dictionary with pages and metadata
        """
        file_path = Path(file_path)
        actual_file_name = file_name or file_path.name

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(str(file_path))

        pages_content = []
        has_images = False

        for page_num in range(len(doc)):
            page = doc[page_num]
            text, metadata = self.extract_text_from_page(page)
            images = self.extract_images_from_page(page)

            if images:
                has_images = True

            pages_content.append({
                "page_number": page_num + 1,
                "text": text,
                "metadata": metadata,
                "images": images,
            })

        doc.close()

        result = {
            "file_name": actual_file_name,
            "file_path": str(file_path),
            "total_pages": len(pages_content),
            "pages": pages_content,
            "has_images": has_images,
            "upload_date": datetime.now().isoformat(),
        }

        return result

    def process_pdf_incremental(
        self,
        file_path: str,
        last_processed_page: int = 0,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Incrementally process a PDF to handle large files.

        Args:
            file_path: Path to the PDF file
            last_processed_page: Last page already processed
            max_pages: Maximum pages to process in this batch

        Returns:
            Dictionary with processed pages and next page to process
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(str(file_path))
        total_pages = len(doc)

        end_page = min(last_processed_page + max_pages, total_pages)

        pages_content = []
        for page_num in range(last_processed_page, end_page):
            page = doc[page_num]
            text, metadata = self.extract_text_from_page(page)
            images = self.extract_images_from_page(page)

            pages_content.append({
                "page_number": page_num + 1,
                "text": text,
                "metadata": metadata,
                "images": images,
            })

        doc.close()

        return {
            "pages": pages_content,
            "total_pages": total_pages,
            "next_page": end_page if end_page < total_pages else None,
            "is_complete": end_page >= total_pages,
        }

    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic PDF information without full processing"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(str(file_path))

        info = {
            "file_name": file_path.name,
            "total_pages": len(doc),
            "metadata": doc.metadata,
        }

        doc.close()
        return info


def clean_text(text: str) -> str:
    """Clean extracted text for better processing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


pdf_processor = PDFProcessor()
