# PHASE 1: Imports and Initial Setup
# ==================================
# Standard library imports
import os
from pathlib import Path
import json # For server info resource

# Third-party imports
# Core MCP
from mcp.server.fastmcp import FastMCP

# LangChain core and community components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
import datetime
# Web Scraping
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# Environment variable loading (optional but recommended)
# from dotenv import load_dotenv
# load_dotenv() # Uncomment if using a .env file
load_dotenv() # Load environment variables from .env file if present
# --- Configuration ---
# API Key Handling (Crucial)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # In a real app, you might raise an error or exit, but for MCP tool
    # it might be better to let tools fail individually if key is needed.
    # However, embeddings NEED the key, so we raise error here.
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

# Path Management
try:
    _default_base_path = Path("./mcp_data/").resolve() # Default path in current dir
    BASE_PATH_STR = os.getenv("MCP_BASE_PATH", str(_default_base_path))
    BASE_PATH = Path(BASE_PATH_STR).resolve()
except Exception as e:
    raise ValueError(f"Invalid BASE_PATH configured: {BASE_PATH_STR}. Error: {e}")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = BASE_PATH / "chroma_db_gemini"
CHROMA_COLLECTION_NAME = "langgraph_gemini_docs"

# LangGraph Docs Path (Assumes file is in BASE_PATH)
LANGGRAPH_DOCS_PATH = BASE_PATH / "llms_full.txt"

# Embedding Model Choice
GEMINI_EMBEDDING_MODEL = "models/embedding-001"


# PHASE 2: MCP Server Initialization and Core Components
# =====================================================

# Create MCP server instance
mcp = FastMCP("Gemini-Chroma-MultiTool-MCP-Server-NoLog")


# Initialize Gemini Embeddings
try:
    # Note: The library uses the GOOGLE_API_KEY environment variable automatically.
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    # Perform a small test embed to verify API key and model access early
    _ = gemini_embeddings.embed_query("initialization_test")
except Exception as e:
    # No logging, raise error to prevent server start with bad config
    raise RuntimeError(f"Fatal Error: Failed to initialize Gemini Embeddings: {e}")

# ChromaDB Client Initialization (Helper function for consistency)
def get_chroma_client():
    """Initializes and returns a Chroma client instance."""
    try:
        # Ensure directory exists (Chroma might handle this, but doesn't hurt)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=gemini_embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR) # Chroma expects string path
        )
        return client
    except Exception as e:
        # Return None, tools need to handle this failure
        print(f"[Error] Failed to initialize Chroma client: {e}") # Print error if no logging
        return None

# PHASE 3: Tool Definitions
# =========================

# Tool 1: RAG Query Tool
@mcp.tool()
def langgraph_rag_query_tool(query: str, k: int = 3) -> str:
    """
    Query the indexed LangGraph documentation using ChromaDB and Gemini embeddings.
    Retrieves k (default 3) most relevant document chunks.

    Args:
        query (str): The query to search the documentation with.
        k (int): The number of documents to retrieve.

    Returns:
        str: A formatted string of the retrieved documents or an error message.
    """
    if not query or not isinstance(query, str):
        return "Error: Query cannot be empty."
    if not isinstance(k, int) or k <= 0:
        # Silently fix invalid k to default without logging
        k = 3

    try:
        chroma_client = get_chroma_client()
        if chroma_client is None:
            return "Error: Failed to initialize ChromaDB client for RAG query."

        retriever = chroma_client.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return f"No relevant documents found in collection '{CHROMA_COLLECTION_NAME}' for query: '{query}'. Database might be empty or query is irrelevant."

        # Format output
        formatted_context = "\n\n".join([f"==DOCUMENT {i+1} (Source: {doc.metadata.get('source', 'Unknown')})==\n{doc.page_content}"
                                         for i, doc in enumerate(relevant_docs)])
        return formatted_context

    except Exception as e:
        # No logging, return error string
        return f"Error processing RAG query: {str(e)}"

# Tool 2: Document Indexing Tool
@mcp.tool()
def index_document_tool(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
    """
    Loads a text document from a file path, splits it, generates Gemini embeddings,
    and indexes them into ChromaDB. Path should be accessible by the server.

    Args:
        file_path (str): Path (absolute or relative to BASE_PATH) of the text file.
        chunk_size (int): Maximum size of text chunks.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        str: Success message indicating chunks indexed or an error message.
    """
    try:
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = (BASE_PATH / file_path).resolve()
        # Basic check for path resolution, more checks below

    except Exception as e:
        return f"Error resolving file path '{file_path}': {str(e)}"

    if not full_path.is_file():
        return f"Error: File not found or is not a regular file at '{full_path}'"

    try:
        # Load the document
        loader = TextLoader(str(full_path), encoding='utf-8')
        documents = loader.load()

        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)

        if not texts:
            return f"Warning: No text content found or split from file '{full_path}'."

        # Index the chunks into ChromaDB
        chroma_client = get_chroma_client()
        if chroma_client is None:
             return "Error: Failed to initialize ChromaDB client for indexing."

        # Simple unique ID generation to allow adding same file content multiple times if needed
        # A more robust approach might check for existing content or use content hashes.
        ids = [f"{full_path}_{i}_{os.urandom(4).hex()}" for i in range(len(texts))]

        # Add documents to the existing collection
        chroma_client.add_documents(texts, ids=ids)

        # Chroma typically persists automatically based on its configuration.
        # Explicit persist call might be needed depending on setup, but usually not.
        # chroma_client.persist()

        return f"Success: Indexed {len(texts)} chunks from '{full_path}'."

    except Exception as e:
        return f"Error indexing file '{full_path}': {str(e)}"

# Tool 3: Web Search Tool (DuckDuckGo)
@mcp.tool()
def web_search_tool(query: str) -> str:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results as a string, or an error message.
    """
    if not query or not isinstance(query, str):
        return "Error: Search query cannot be empty."

    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except ImportError:
        return "Error: DuckDuckGo search dependency not installed. Please install 'duckduckgo-search'."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Tool 4: Web Scraper Tool (Basic)
@mcp.tool()
def web_scrape_tool(url: str, timeout: int = 10) -> str:
    """
    Fetches URL content, extracts text using BeautifulSoup. Basic error handling.
    No JavaScript execution. Identifies itself politely with User-Agent.

    Args:
        url (str): The URL to scrape (must start with http:// or https://).
        timeout (int): Request timeout in seconds (default: 10).

    Returns:
        str: Extracted text content or an error message.
    """
    if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
        return "Error: Invalid URL provided. Must start with http:// or https://."

    headers = {
        'User-Agent': 'MCPExampleBot/1.0' # Identify your bot
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            return f"Error: URL did not return HTML content (Content-Type: {content_type})."

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text, trying to be cleaner
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, use newline as separator, strip leading/trailing whitespace from lines
        text = soup.get_text(separator='\n', strip=True)

        # Remove excessive blank lines
        lines = (line.strip() for line in text.splitlines())
        cleaned_text = '\n'.join(line for line in lines if line)

        if not cleaned_text:
            return f"Warning: No text content extracted from URL: {url}"

        return cleaned_text

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout} seconds for URL: {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL '{url}': {str(e)}"
    except ImportError:
         return "Error: Web scraping dependencies ('requests', 'beautifulsoup4') not installed."
    except Exception as e:
        return f"Error scraping URL '{url}': {str(e)}"

# Tool 5: Simple File Reader Tool
@mcp.tool()
def read_text_file_tool(file_path: str) -> str:
    """
    Reads content of a local text file. Security Warning: Allows reading any file
    accessible by the server process. Use with extreme caution.

    Args:
        file_path (str): Absolute path or path relative to BASE_PATH of the text file.

    Returns:
        str: Content of the file or an error message.
    """
    try:
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = (BASE_PATH / file_path).resolve()
        # Security check happens implicitly via filesystem permissions mostly
        # Consider adding explicit checks here if needed based on BASE_PATH

    except Exception as e:
        return f"Error resolving file path '{file_path}': {str(e)}"

    if not full_path.is_file():
        return f"Error: File not found or is not a regular file at '{full_path}'"

    # Add security warning print to console even without logging
    print(f"[Security Warning] read_text_file_tool attempting to access: {full_path}")

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        # This check is somewhat redundant given the is_file() check above, but safe
        return f"Error: File not found at '{full_path}'."
    except PermissionError:
        return f"Error: Permission denied while reading file '{full_path}'."
    except UnicodeDecodeError:
        return f"Error: Could not decode file '{full_path}' using UTF-8 encoding. It might be binary or use a different text encoding."
    except Exception as e:
        return f"Error reading file '{full_path}': {str(e)}"

# Tool 6: Get Date and Time Tool
@mcp.tool()
def get_datetime_tool() -> str:
    """
    Returns the current date and time as a formatted string.

    Returns:
        str: Current date and time or an error message.
    """
    try:
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error getting current date and time: {str(e)}"
    
    
# Tool 7: File write tool
@mcp.tool()
def write_text_file_tool(file_path: str, content: str) -> str:
    """
    Writes content to a local text file. Security Warning: Allows writing any file
    accessible by the server process. Use with extreme caution.

    Args:
        file_path (str): Absolute path or path relative to BASE_PATH of the text file.
        content (str): Content to write to the file.

    Returns:
        str: Success message or an error message.
    """
    try:
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = (BASE_PATH / file_path).resolve()
        # Security check happens implicitly via filesystem permissions mostly
        # Consider adding explicit checks here if needed based on BASE_PATH

    except Exception as e:
        return f"Error resolving file path '{file_path}': {str(e)}"

    # Add security warning print to console even without logging
    print(f"[Security Warning] write_text_file_tool attempting to access: {full_path}")

    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Success: Content written to file '{full_path}'."
    except PermissionError:
        return f"Error: Permission denied while writing to file '{full_path}'."
    except Exception as e:
        return f"Error writing to file '{full_path}': {str(e)}"
    
    
# Tool 8: Folder and file listing tool
@mcp.tool()
def list_files_tool() -> str:
    """
    Lists all files and folders in the current working directory.

    Returns:
        str: List of files and folders or an error message.
    """
    try:
        # Get the current working directory
        current_directory = os.getcwd()
        # List all files and folders in the current directory
        files_and_folders = os.listdir(current_directory)
        # Format the list as a string
        formatted_list = "\n".join(files_and_folders)
        return f"Files and folders in '{current_directory}':\n{formatted_list}"
    except Exception as e:
        return f"Error listing files: {str(e)}"    


# PHASE 4: Resource Definitions
# =============================

# Resource 1: Full LangGraph Documentation
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get the full LangGraph documentation content from the pre-configured file path.

    Returns:
        str: Content of the documentation file or an error message.
    """
    try:
        if not LANGGRAPH_DOCS_PATH.is_file():
             return f"Error: LangGraph documentation file not found at '{LANGGRAPH_DOCS_PATH}'"

        with open(LANGGRAPH_DOCS_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading documentation file '{LANGGRAPH_DOCS_PATH}': {str(e)}"

# Resource 2: Example Server Configuration Resource
@mcp.resource("config://server/info")
def get_server_info() -> str:
    """
    Provides basic configuration information about this MCP server instance.
    Excludes sensitive information like API keys, but lists configured tools and resources.
    """
    # Safely grab the private registries (fall back to empty dicts if not present)
    tools_registry     = getattr(mcp, "_tools", {})     # dict: tool_name -> function
    resources_registry = getattr(mcp, "_resources", {}) # dict: uri_pattern -> function

    info = {
        "server_name": mcp.name,
        "base_path": str(BASE_PATH),
        "chroma_persist_directory": str(CHROMA_PERSIST_DIR),
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "langgraph_docs_path": str(LANGGRAPH_DOCS_PATH),
        # List out the registered tool names and resource URIs:
        "available_tools": sorted(tools_registry.keys()),
        "available_resources": sorted(resources_registry.keys()),
    }
    # Return as pretty-printed JSON
    return json.dumps(info, indent=2)


# PHASE 5: Running the Server
# ===========================

if __name__ == "__main__":
    print(f"--- Starting MCP Server: {mcp.name} ---")
    print(f"Base Path: {BASE_PATH}")
    print(f"ChromaDB Path: {CHROMA_PERSIST_DIR}")
    print(f"Chroma Collection: {CHROMA_COLLECTION_NAME}")
    print(f"LangGraph Docs Path: {LANGGRAPH_DOCS_PATH}")
    print("--- Server Initialization Complete ---")
    print(f"Running MCP server with 'stdio' transport. Listening on standard input...")
    # To run with TCP transport (e.g., on port 8000):
    # Change transport='tcp', add host='0.0.0.0', port=8000
    # Requires: pip install fastapi uvicorn
    # mcp.run(transport='tcp', host='0.0.0.0', port=8000)
    mcp.run(transport='stdio')