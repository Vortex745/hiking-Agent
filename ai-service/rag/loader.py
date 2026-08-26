import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.text_processing import denoise_text, normalize_text


class DocumentLoader:
    """Load and split documents for RAG processing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> str:
        """Load file content based on extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext == ".txt":
            return path.read_text("utf-8")
        elif ext == ".md":
            return path.read_text("utf-8")
        elif ext == ".pdf":
            # For PDF, try basic text extraction
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                return f"[注意: pdfplumber 未安装，无法解析 PDF 文件 {path.name}]"
        elif ext == ".docx":
            return self._load_docx(path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}（支持: .txt, .md, .pdf, .docx）")

    def _load_docx(self, path: Path) -> str:
        """Extract paragraph text from a docx file with the standard library."""
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)

    def split(self, text: str) -> list[Document]:
        """Split text into chunks with paragraph-first, recursive fallback strategy."""
        chunks: list[Document] = []
        sections = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]

        for section in sections:
            if len(section) <= self.splitter._chunk_size:
                chunks.append(Document(page_content=section))
                continue
            chunks.extend(self.splitter.create_documents([section]))

        if not chunks and text.strip():
            return self.splitter.create_documents([text])

        return chunks

    def load_and_split(self, file_path: str) -> list[Document]:
        """Load and split a file in one call."""
        raw_text = self.load_file(file_path)
        unified_text = normalize_text(raw_text)
        cleaned_text = denoise_text(unified_text)
        docs = self.split(cleaned_text)

        path = Path(file_path)
        title = self._extract_title(unified_text, path)
        chunk_count = len(docs)
        for index, doc in enumerate(docs):
            content_hash = hashlib.sha1(f"{path.name}:{index}:{doc.page_content}".encode("utf-8")).hexdigest()
            doc.metadata.update({
                "id": content_hash,
                "source": path.name,
                "file_name": path.name,
                "title": title,
                "doc_type": path.suffix.lower().lstrip("."),
                "chunk_index": index,
                "chunk_count": chunk_count,
                "chunk_strategy": "hybrid_recursive",
                "content_hash": content_hash,
            })
        return docs

    def _extract_title(self, text: str, path: Path) -> str:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^#+\s*", "", line)
            title = denoise_text(line)
            if title:
                return title[:80]
        return path.stem
