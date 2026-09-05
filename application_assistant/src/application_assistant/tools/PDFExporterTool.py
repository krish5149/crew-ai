from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from fpdf import FPDF
import os


class PDFExporterInput(BaseModel):
    content: str = Field(..., description="The full text/markdown content to write into the PDF")
    file_name: str = Field(..., description="Output PDF file name, e.g. 'resume_google.pdf'")


class PDFExporterTool(BaseTool):
    name: str = "PDF Exporter Tool"
    description: str = (
        "Converts plain text or markdown content into a formatted PDF file "
        "and saves it to disk. Use this to export a final resume or cover "
        "letter as a downloadable PDF."
    )
    args_schema: Type[BaseModel] = PDFExporterInput

    # Map common Unicode punctuation (smart quotes, dashes, etc.) that
    # LLMs tend to generate to their closest Latin-1 / ASCII equivalents,
    # since the core PDF fonts only support Latin-1 encoding.
    _UNICODE_REPLACEMENTS = {
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "-",   # bullet
        "\u00a0": " ",   # non-breaking space
    }

    @classmethod
    def _sanitize(cls, text: str) -> str:
        for unicode_char, replacement in cls._UNICODE_REPLACEMENTS.items():
            text = text.replace(unicode_char, replacement)
        # Fallback: drop any remaining characters the core font can't encode
        return text.encode("latin-1", "ignore").decode("latin-1")

    def _run(self, content: str, file_name: str) -> str:
        content = self._sanitize(content)
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        output_path = os.path.join(output_dir, file_name)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        for line in content.splitlines():
            line = line.strip()
            if not line:
                pdf.ln(4)
                continue
            # Simple markdown heading/bold handling
            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 15)
                pdf.multi_cell(0, 8, line.lstrip("# ").strip())
                pdf.set_font("Helvetica", size=11)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 7, line.lstrip("# ").strip())
                pdf.set_font("Helvetica", size=11)
            elif line.startswith("- ") or line.startswith("* "):
                pdf.multi_cell(0, 6, f"  - {line[2:].strip()}")
            else:
                pdf.multi_cell(0, 6, line)

        pdf.output(output_path)
        return f"PDF successfully saved at: {output_path}"
