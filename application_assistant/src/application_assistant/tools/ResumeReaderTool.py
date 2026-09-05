from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pypdf import PdfReader
from docx import Document
import os

class ResumeReaderInput(BaseModel):
    file_path: str = Field(..., description="Path to the resume file (.pdf or .docx).")


class ResumeReaderTool(BaseTool):
    name: str = "Resume Reader Tool"
    description: str = (
        "Reads a candidate's resume from a PDF or DOCX file and returns its "
        "plain text content for analysis and tailoring."
    )
    args_schema: Type[BaseModel] = ResumeReaderInput

    def _run(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext == ".docx":
            return self._read_docx(file_path)
        else:
            return f"Unsupported file type: {ext}. Please provide a .pdf or .docx file."

    def _read_pdf(self,file_path: str) -> str:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()

    def _read_docx(self,file_path: str) -> str:
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text.strip()

        
