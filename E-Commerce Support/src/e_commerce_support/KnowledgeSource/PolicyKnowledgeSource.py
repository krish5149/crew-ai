from pathlib import Path
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

def LoadKnowledge():
    return TextFileKnowledgeSource(
        file_paths=[Path("/workspaces/bles/E-Commerce Support/knowledge/returnpolicy.txt")]
    )
