import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    # The query must be a dictionary containing a 'query' key with the query string
    query: dict = Field(..., description="Query to search the document. Must contain a 'query' key with the query string.")

class DocumentSearchTool(BaseTool):
    # Tool name and description
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query string."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    # Allow extra fields in the model config
    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """
        Initialize the searcher with a PDF file path and set up the Qdrant collection.
        """
        super().__init__()
        self.file_path = file_path
        # Use in-memory Qdrant client for small experiments
        self.client = QdrantClient(":memory:")
        # Process the document and add its chunks to the vector store
        self._process_document()

    def _extract_text(self) -> str:
        """
        Extract raw text from PDF using MarkItDown.
        """
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """
        Split the raw text into smaller chunks for efficient search.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        # Wrap each chunk string in a dictionary with a 'text' key
        return [{"text": chunk} for chunk in chunks]

    def _process_document(self):
        """
        Process the document: extract text, split into chunks, and add to Qdrant collection.
        """
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)

        # Extract the text from each chunk dictionary
        docs = [chunk['text'] for chunk in chunks] 
        
        # Prepare metadata and IDs for each chunk
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        # Add the document chunks to the Qdrant collection
        self.client.add(
            collection_name="demo_collection",
            documents=docs, # Use docs instead of chunks
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: dict) -> list:
        """
        Search the document with a query string and return relevant chunks.
        """
        query = query['query']
        # Query the Qdrant collection for relevant chunks
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        # Extract the document text from each relevant chunk
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        # Join the relevant chunks with a separator and return
        return separator.join(docs)