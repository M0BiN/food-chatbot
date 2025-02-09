from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Load environment variables
load_dotenv()

# Constants
PDF_FILE_PATH = './The New Complete Book of Foos.pdf'
LANCE_DB_PATH = './lancedb'
TABLE_NAME = "food"
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Functions
def parse_pdf(file_path):
    """Parse a PDF file into documents."""
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}
    return SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

def split_documents(documents, chunk_size, chunk_overlap):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_chunks = []
    for source in documents:
        for chunk in splitter.split_text(source.get_content()):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks

def initialize_embedder(model_name):
    """Initialize the embedding model."""
    return get_registry().get("sentence-transformers").create(name=model_name)

def prepare_data(source_chunks):
    """Prepare data for LanceDB."""
    return [
        {
            "text": source_chunks[idx].page_content,
            "id": idx + 1
        }
        for idx in range(len(source_chunks))
    ]

def create_lancedb_table(db_path, table_name, schema, data):
    """Create a table in LanceDB and add data."""
    db = lancedb.connect(db_path)
    tbl = db.create_table(table_name, schema=schema, mode="overwrite")
    tbl.add(data)
    return tbl

def run_hybrid_search(table):
    """Create a full-text search index for hybrid search."""
    table.create_fts_index("text", replace=True)

# Main Logic
def main():
    # Parse the PDF file
    documents = parse_pdf(PDF_FILE_PATH)

    # Split documents into smaller chunks
    source_chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # Initialize the embedder
    embedder = initialize_embedder(EMBEDDING_MODEL_NAME)

    # Define the schema
    class Schema(LanceModel):
        id: int
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()

    # Prepare data for insertion into LanceDB
    data = prepare_data(source_chunks)

    # Create LanceDB table and add data
    table = create_lancedb_table(LANCE_DB_PATH, TABLE_NAME, Schema, data)
    print(f"Inserted {len(data)} records into LanceDB.")

    # Run hybrid search with a reranker
    run_hybrid_search(table)

# Run the script
if __name__ == "__main__":
    main()
