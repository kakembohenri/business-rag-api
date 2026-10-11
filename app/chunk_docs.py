import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SECTION_RE  = re.compile(r"^(\d{1,2}\.\s+\S.*)$", re.MULTILINE)   # "7. Distribution Locations"
QUESTION_RE = re.compile(r"^(Q\d+\.\s.*)$", re.MULTILINE)          # FAQ: "Q12. How long ...?"
PAGE_MARK   = re.compile(r"@@PAGE:(\d+)@@\n?")
TRAILING_MARKS = re.compile(r"(?:@@PAGE:\d+@@\s*)+$")

def add_markdown_headings(text: str) -> str:
    """The PDFs use plain-text headings; turn them into real markdown headings so the splitter can see them."""
    text = text.replace("<br>", " ").replace("**", "")              # PyMuPDF table-cell markup is noise for embeddings
    text = SECTION_RE.sub(r"## \1", text)
    return QUESTION_RE.sub(r"### \1", text)

def chunk_documents(pages: list[Document]) -> list[Document]:
    """Pass ONE PDF's page Documents (in order). Pages are merged first, so a section that crosses a page
    break (heading on page 2, table on page 3) stays in one chunk. Page markers track where each chunk starts."""
    title = pages[0].metadata["title"]
    text = "\n".join(f"@@PAGE:{p.metadata['page']}@@\n{add_markdown_headings(p.page_content)}" for p in pages)

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section"), ("###", "question")], strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])

    running_page, final = pages[0].metadata["page"], []
    for chunk in text_splitter.split_documents(md_splitter.split_text(text)):
        raw = chunk.page_content
        marks = [int(m) for m in PAGE_MARK.findall(raw)]
        starts_with_mark = bool(PAGE_MARK.match(raw.lstrip()))
        start = marks[0] if (marks and starts_with_mark) else running_page
        content_marks = [int(m) for m in PAGE_MARK.findall(TRAILING_MARKS.sub("", raw))]
        end = max([start] + content_marks)                     # a trailing marker has no content of its own
        if marks:
            running_page = marks[-1]                           # next chunk starts on the last page seen

        body = PAGE_MARK.sub("", raw).strip()
        if not body:
            continue
        section = chunk.metadata.get("section", "Introduction")  # text before the first numbered heading
        final.append(Document(
            page_content=f"[{title} | {section} | Page {start}]\n{body}",
            metadata={"title": pages[0].metadata["title"], "page": start, "page_end": end,
                      "section": section, "has_tables": "|---|" in body}))
    return final