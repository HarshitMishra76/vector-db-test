import json
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

COLLECTION_NAME = "vector-db-test"


#
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "vector-db-test"
# os.environ["USER_AGENT"] = "vector-db-test-project"

def create_collection():
    try:
        get_qdrant_client().create_collection(collection_name=COLLECTION_NAME,
                                              vectors_config=models.VectorParams(size=1024,
                                                                                 distance=models.Distance.COSINE))
    except Exception as e:
        print(f"While creating collection ,Error is {e}")


def query():
    contexts = []
    question: list[str] = [
        "What are the key differences in data models between MongoDB and PostgreSQL?",
        "How do MongoDB and PostgreSQL handle unstructured and semi-structured data?",
        "Which database system, MongoDB or PostgreSQL, is more scalable, and why?",
        "What are the typical use cases for MongoDB and PostgreSQL?",
        "How do MongoDB and PostgreSQL perform in read-heavy and write-heavy workloads?",
        "What are the differences in indexing mechanisms between MongoDB and PostgreSQL?",
        "How do MongoDB and PostgreSQL ensure ACID compliance?",
        "What are the challenges associated with data migration between MongoDB and PostgreSQL?",
        "Which database offers more flexibility in schema design?",
        "What consistency model is used by PostgreSQL and MongoDB?",
        "How well do MongoDB and PostgreSQL integrate with other technologies?",
        "What are the licensing and cost models for MongoDB and PostgreSQL?",
        "What kind of community support and ecosystem exist for MongoDB and PostgreSQL?",
        "What are the performance tuning techniques for MongoDB and PostgreSQL?",
        "How do MongoDB and PostgreSQL handle backups and disaster recovery?",
        "How to use useEffect in React?",
        "How to use collections in PostgreSQL?",
        "How is MongoDB different from MySQL?"
    ]

    for q in question:
        c: list[str] = []
        result: list[Document] = vectorstore.max_marginal_relevance_search(q, 4)
        for doc in result:
            c.append(doc.page_content)
        contexts.append(c)
    file = Path("sample.json")
    file.touch()
    file.write_text(
        json.dumps({
            "question": question,
            "contexts": contexts
        })
    )


def delete_all():
    print("deleting the collection...")
    get_chromadb_client().delete_collection(COLLECTION_NAME)
    # get_qdrant_client().delete_collection(COLLECTION_NAME)
    print("Collection deleted")


def embedd_all():
    print("Start embedding ...")
    references = Path("references")
    all_docs = []

    for file in references.iterdir():
        if file.is_file() and file.suffix.endswith(".pdf"):
            loader = PyPDFLoader(file)
            documents = loader.load()
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                docs: list[Document] = text_splitter.split_documents(documents)
            except Exception as e:
                print(f"Something went wrong while splitting {file.name}")
                print(f"Error is {e}")
                continue

            for doc in docs:
                doc.metadata = {
                    "file_name": file.name,
                }
            try:
                all_docs += docs
                vectorstore.add_documents(documents=all_docs)
            except Exception as e:
                print(f"Something went wrong while adding  {file.name}")
                print(f"Error is {e}")

    print("Completed Embedding")


def get_chromadb_client():
    client = chromadb.HttpClient(host="localhost", port="8005")
    return client


def get_qdrant_client():
    q_client = QdrantClient(
        url="http://localhost:6333"
    )
    return q_client


if __name__ == '__main__':

    vectorstore = Chroma(
        client=get_chromadb_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceBgeEmbeddings(),
    )

    # vectorstore = Qdrant(
    #     client=get_qdrant_client(),
    #     collection_name=COLLECTION_NAME,
    #     embeddings=HuggingFaceBgeEmbeddings()
    # )

    while True:
        user_input = input("""
Enter a number (or -1 to exit):
1. To query a question
2. To embedd the documents
3. To delete the collection
4. To create collection
""")

        if user_input == "-1":
            break

        try:
            num = int(user_input)

            if num == 1:
                query()
            elif num == 2:
                # execute function for 2
                embedd_all()
            elif num == 3:
                # execute function for 3
                delete_all()
            elif num == 4:
                # execute function for 3
                create_collection()
            else:
                print("Invalid input. Please enter a number between 1 and 4, or -1 to exit.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    pass
