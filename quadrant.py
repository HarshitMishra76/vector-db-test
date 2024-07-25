import pathlib
from pathlib import Path
from typing import Any, List

from chromadb.api import ClientAPI
from fastapi import HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_community.vectorstores import Qdrant


REFERENCES_PATH = Path("files", "references")

file_keyword_map: dict[str, list[str]] = {
    "Building a search engine for interactive search.txt": ["Kubernetes", "Docker", "Cube ADM", "API server",
                                                            "Static pod", "EC2 instance", "kubelet", "Manifest file",
                                                            "Control plane", "Terraform", "CRI Docker",
                                                            "Container runtime", "Configuration", "Workload",
                                                            "Infrastructure", "Debugging", "Node setup", "Pod subnet",
                                                            "Cluster setup", "kubectl"],
    "Data Science in e-Commerce.txt": ["Search Engine", "Interactive Search", "TypeSense", "Latency", "ElasticSearch",
                                       "Indexing", "Query Performance", "Autocomplete", "Scalability",
                                       "Search Infrastructure", "User Experience", "Speed Optimization", "Metadata",
                                       "API Integration", "Data Indexing", "Real-Time Search", "Performance Metrics",
                                       "Network Latency", "Query Ranking", "Cloud Hosting"],
    "LLMs in action.txt": ["HTTP Server", "Data Store", "Latency", "File System", "Disk I/O", "Persistence",
                           "Performance", "Concurrency", "Cache", "Read/Write Operations", "Request Handling",
                           "Data Persistence", "Memory Management", "System Architecture", "Benchmarking",
                           "Disk Storage", "Operating System", "Service Layer", "Data Update", "File Access"],
    "Uncover the mysteries of Infrastructure as code.txt": ["Data Science", "Recommendation Engine",
                                                            "Predictive Analytics", "Delivery SLA",
                                                            "Sentiment Analysis", "Order Fulfillment",
                                                            "Dynamic Pricing", "User Review Aggregation",
                                                            "Catalog Normalization", "Data Warehousing",
                                                            "API Integration", "Conversion Optimization",
                                                            "Inventory Forecasting", "Return Merchandise Authorization",
                                                            "Personalized Marketing", "Anomaly Detection",
                                                            "Supply Chain Management", "Demand Forecasting",
                                                            "Multi-channel Engagement", "Feedback Loop"],
    "Building a write intensive DB and impact of different file systems on it.txt": ["Large Language Models",
                                                                                     "Retrieval-Augmented Generation",
                                                                                     "Embedding Extraction",
                                                                                     "Semantic Search", "Text Chunking",
                                                                                     "Contextual Prompting",
                                                                                     "Generative Modeling",
                                                                                     "Probabilistic Modeling",
                                                                                     "Inference Theory",
                                                                                     "Hallucination",
                                                                                     "Data Augmentation",
                                                                                     "Knowledge Representation",
                                                                                     "Out-of-Scope Responses",
                                                                                     "Sentiment Analysis",
                                                                                     "Model Fine-Tuning",
                                                                                     "Interactive Prompting",
                                                                                     "Dynamic Querying",
                                                                                     "Vector Similarity Search",
                                                                                     "Conversational AI",
                                                                                     "Pre-trained Models"],
    "Shipping LLM Addressing Production Challenges.txt": ["LLM", "Production Challenges", "Document Processing",
                                                          "Information Extraction", "Generative AI", "Knowledge Base",
                                                          "Question Answering", "Conversational Agents",
                                                          "Workflow Automation", "Retrieval-Augmented Generation",
                                                          "Embedding Extraction", "Semantic Search",
                                                          "Data Retrieval Pipeline", "Data Synthesis Pipeline",
                                                          "Model Accuracy", "Performance Tracking",
                                                          "Pipeline Evaluation", "Contextual Relevance",
                                                          "Cost and Latency", "Scalability"],
    "Understanding the internal workings of databases.txt": ["Infrastructure as Code", "Configuration Management",
                                                             "Infrastructure Evolution", "Cloud Computing",
                                                             "Manual Configuration", "Scripting",
                                                             "Declarative Approach", "Configuration Drift", "Chef",
                                                             "Puppet", "Ansible", "Server Management",
                                                             "Virtual Machines", "Automation Tools",
                                                             "Imperative Scripting", "Scaling Challenges",
                                                             "State Management", "Desired State", "Automation Scripts",
                                                             "Tool Adoption"],
    "Bootstrapping Kubernetes.txt": ["Kubernetes", "Databases", "Indexing", "Data Files", "Clustered Index",
                                     "Non-Clustered Index", "Data Structures", "Transaction Logging",
                                     "In-Memory Databases", "Index Organized Tables", "Heap Organized Tables",
                                     "SQL Queries", "Database Optimization", "Data Storage", "Database Fundamentals",
                                     "Mechanical Sympathy", "High-Scale Systems", "Embedded Databases",
                                     "Sequential Reads", "Random Writes", "Durability"]
}


def store_embeddings(collection_id: str, docs: list[Document],
                     embedding_function: Embeddings) -> None:
    """Return None. Stores embedding to the vector store."""
    url = "http://localhost:6333"

    q_client = QdrantClient(
        url=url
    )

    q_client.create_collection(collection_name=collection_id,vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE))
    qdrant = Qdrant.from_documents(
        docs,
        embedding_function,
        url=url,
        prefer_grpc=True,
        collection_name=collection_id,
    )

    q_client.create_payload_index(
        collection_name=collection_id,
        field_name="keywords",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=15,
            lowercase=True,
        ),
)


def create_embeddings(files: List[Path], collection_id: str) -> None:
    """Return None. Create embedding of documents for given collection_id."""

    url = "http://localhost:6333"


    for file in files:
        try:
            loader = TextLoader(REFERENCES_PATH / file.name)
            documents = loader.load()
            for doc in documents:
                keywords = ', '.join(file_keyword_map[file.name])
                metadata = {
                    "keywords": keywords,
                    "filename": file.name
                }
                doc.metadata = metadata
        except Exception as e:
            raise HTTPException(status_code=415, detail=str(e))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                       chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        store_embeddings(collection_id, docs, HuggingFaceBgeEmbeddings())
    #
    # q_client.create_payload_index(
    #     collection_name=collection_id,
    #     field_name="keywords",
    #     field_schema=models.TextIndexParams(
    #         type="text",
    #         tokenizer=models.TokenizerType.WORD,
    #         min_token_len=2,
    #         max_token_len=15,
    #         lowercase=True,
    #     ),
    # )



files = [
    Path("Building a search engine for interactive search.txt"),
    Path("Data Science in e-Commerce.txt"),
    Path("LLMs in action.txt"),
    Path("Uncover the mysteries of Infrastructure as code.txt"),
    Path("Building a write intensive DB and impact of different file systems on it.txt"),
    Path("Shipping LLM Addressing Production Challenges.txt"),
    Path("Understanding the internal workings of databases.txt"),
    Path("Bootstrapping Kubernetes.txt"),
]

collection_id = "documents"

# create_embeddings(files, collection_id)


def query(query_):
    url = "http://localhost:6333"
    q_client = QdrantClient(
        url=url
    )

    qdrant = Qdrant(
        client=q_client,
        collection_name=collection_id,
        embeddings=HuggingFaceBgeEmbeddings()
    )

    # keyword_filter=models.Filter(
    #     should=[
    #         models.FieldCondition(
    #             key="keywords",
    #             match=models.MatchAny(value="Kubernetes")
    #         )
    #     ]
    # )

    keyword_filter = models.Filter(
        should=[
            models.FieldCondition(
                key="keywords",
                match=models.MatchText(text="Data Science")
            )
        ]
    )

    found_docs = qdrant.similarity_search_with_relevance_scores(query_, filter=keyword_filter)
    found_docs1 = qdrant.similarity_search_with_relevance_scores(query_)
    print(found_docs1)

    return found_docs

ans = query("Can you refer some good techniques about deployment?")
for a in ans:
    print(a)