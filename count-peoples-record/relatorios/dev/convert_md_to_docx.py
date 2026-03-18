import re
import os
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def convert_md_to_docx(md_path, docx_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue

        # Headers
        if line.startswith('# '):
            p = doc.add_heading(level=0)
            process_formatting(p, line[2:])
            i += 1
        elif line.startswith('## '):
            p = doc.add_heading(level=1)
            process_formatting(p, line[3:])
            i += 1
        elif line.startswith('### '):
            p = doc.add_heading(level=2)
            process_formatting(p, line[4:])
            i += 1
        
        # Tables
        elif line.startswith('|'):
            # Find table end
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            # Simple table parsing: remove the divider line if it looks like |---|---|
            data_lines = []
            for l in table_lines:
                # Basic check for divider: mostly | - :
                if re.match(r'^\|[\s:-|]+\|$', l):
                    continue
                data_lines.append(l)
                
            if data_lines:
                rows = []
                for l in data_lines:
                    # Capture everything between pipes
                    cells = [c.strip() for c in l.split('|')][1:-1]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    num_cols = max(len(r) for r in rows)
                    table = doc.add_table(rows=0, cols=num_cols)
                    table.style = 'Table Grid'
                    for row_data in rows:
                        row_cells = table.add_row().cells
                        for idx, val in enumerate(row_data):
                            if idx < len(row_cells):
                                # Inside table cells, we also want formatting
                                p = row_cells[idx].paragraphs[0]
                                process_formatting(p, val)
            continue

        # Lists
        elif line.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            # Handle formatting in list items
            process_formatting(p, line[2:])
            i += 1
        
        # Paragraphs
        else:
            p = doc.add_paragraph()
            process_formatting(p, line)
            i += 1

    doc.save(docx_path)
    print(f"Successfully converted {md_path} to {docx_path}")

def process_formatting(paragraph, text):
    # Basic formatter for bold and italic text
    # Sequence: bold (**), then italic (*)
    # For now, let's keep it simple: **text** for bold, *text* for italic if needed
    
    # We use a greedy regex-based approach or split carefully
    # Splitting by bold tags first
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            content = part[2:-2]
            # Check for italic within bold? Maybe later.
            run = paragraph.add_run(content)
            run.bold = True
        else:
            # Check for italic in the remaining text
            subparts = re.split(r'(\*.*?\*)', part)
            for subpart in subparts:
                if subpart.startswith('*') and subpart.endswith('*'):
                    run = paragraph.add_run(subpart[1:-1])
                    run.italic = True
                else:
                    paragraph.add_run(subpart)

if __name__ == "__main__":
    md_file = r"c:\Users\patrickcruz\Documents\Professional\Github\contagem-de-pessoas\count-peoples-record\relatorios\relatorio_contagem_pessoas.md"
    docx_file = r"c:\Users\patrickcruz\Documents\Professional\Github\contagem-de-pessoas\count-peoples-record\relatorios\relatorio_contagem_pessoas.docx"
    convert_md_to_docx(md_file, docx_file)
