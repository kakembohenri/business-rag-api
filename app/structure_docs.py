import pymupdf
from langchain_core.documents import Document
from langchain_core.documents import Document
from pathlib import Path

file_path = Path(__file__).parent

def pdf_to_markdown(path: str) -> list[Document]:
    doc = pymupdf.open(path)
    documents = []

    title = doc.metadata.get("title")
    for page_num, page in enumerate(doc):
        raw_text = page.get_text("text").strip()

        if not raw_text:
            continue

        text_lines = raw_text.split("\n")
        if text_lines:
            text_lines.pop(0)

        page_text = "\n".join(text_lines).strip()

        table_finder = page.find_tables()

        if table_finder.tables:
            for table in table_finder.tables:
                # Append the clean Markdown representation
                header_names = getattr(table.header, "names", [])

                if not header_names and table.extract():
                    # Fallback to the first row of matrix data if names attribute is empty
                    header_names = table.extract()[0]

                # 2. Format a clean title from the columns (e.g., "Year / Milestone Table")
                clean_header_title = " / ".join([str(h).strip() for h in header_names if h]).strip()

                if not clean_header_title:
                    clean_header_title = "Structured Data" # Fallback security string

                flat_cell_list = []
                for row in table.extract():
                    for cell in row:
                        if cell and cell.strip():
                            flat_cell_list.append(cell.strip())

                duplicate_text_block = "\n".join(flat_cell_list)

                clean_header_title = "Structured Data"  # Fallback string
                
                markdown_table = f"\n{table.to_markdown()}\n"

                page_text = page_text.replace(duplicate_text_block, markdown_table)

        clean_page_content = "\n".join([line.strip() for line in page_text.split("\n") if line.strip()])

        clean_page_content = clean_page_content.replace("•\n","-") # Replace bullet points with hyphens for better Markdown formatting
        
        if clean_page_content:
            doc_obj = Document(
                page_content=clean_page_content,
                metadata={
                    "title": title,
                    "page": page_num + 1,
                    "has_tables": bool(table_finder.tables)
                }
            )
            documents.append(doc_obj)

    return documents