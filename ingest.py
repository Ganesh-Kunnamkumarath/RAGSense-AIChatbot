import os
from pinecone import Pinecone  # Updated Pinecone import
from langchain_community.embeddings import OpenAIEmbeddings  # Updated LangChain import
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Updated LangChain import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # Updated LangChain import

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "chatbot-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' does not exist. Please create it first.")
    exit()

# Connect to the Pinecone index
index = pc.Index(INDEX_NAME)

# Load and split documents
loader = TextLoader("data.txt")  # Ensure "data.txt" exists in the directory
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a Pinecone vector store and insert documents
vector_store = PineconeVectorStore.from_documents(docs, embedding_model, index_name=INDEX_NAME)

# Check the final vector count
total_vectors = index.describe_index_stats()
print(f"Total vectors in index after insertion: {total_vectors}")
