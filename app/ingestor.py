from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from chunk_docs import chunk_documents
from langchain_openai import OpenAIEmbeddings

from structure_docs import pdf_to_markdown
from config import get_settings

file_path = Path(__file__).parent

settings = get_settings()

BATCH_SIZE = 20  

print("------ Initializing Pinecone Ingestion ------ \n")
# embeddings = OllamaEmbeddings(
#     model=settings.embeddings_model_name
# )

embeddings = OpenAIEmbeddings(
    model=settings.embeddings_model_name,
    api_key=settings.openai_api_key,
    dimensions=1024,
)

pinecone_ingestor = PineconeVectorStore(
    embedding=embeddings, 
    index_name=settings.index_name,
    pinecone_api_key=settings.pinecone_api_key
)

index_stats = pinecone_ingestor.index.describe_index_stats()

total_vectors = index_stats['total_vector_count']

if total_vectors > 0:
    print(f"Index '{settings.index_name}' already has {total_vectors} vectors. Skipping ingestion.")

documents = [
    "company_brochure.pdf",
    "product_catalogue.pdf",
    "customer_faq.pdf",
    "returns_warranty_policy.pdf",
    "shipping_delivery_policy.pdf",
    "supplier_and_procurement_policy.pdf"
]

print("------ Generating markdown documents from PDFs ------ \n")
chunks_to_embed = []
for doc in documents:
    print(f"------ Processing {doc} ------ \n")
    docs = pdf_to_markdown(f"{file_path.parent}/documents/{doc}")
    print(f"------ Chunking {doc} ------ \n")
    chunks = chunk_documents(docs)
    chunks_to_embed.extend(chunks)

print(f"------ Generated {len(chunks_to_embed)} chunks from {len(documents)} documents ------ \n")
print("------ Ingesting Chunks into Pinecone ------ \n")

for i in range(0, len(chunks_to_embed), BATCH_SIZE):
    batch = chunks_to_embed[i : i + BATCH_SIZE]
    
    print(f"Uploading batch {(i // BATCH_SIZE) + 1} (Chunks {i} to {i + len(batch)})...")
    
    pinecone_ingestor.add_documents(batch)

print("\n🎉 Ingestion complete! All documents successfully vectorized and indexed.")