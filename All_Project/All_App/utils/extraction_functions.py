import os
from docx import Document


def All_Extraction(file_path):
    if file_path.lower().endswith('.docx'):
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        print("paragraph", type(paragraphs))
        return paragraphs
    
    