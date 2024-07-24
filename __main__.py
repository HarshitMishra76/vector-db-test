from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "vector-db-test"


#
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "vector-db-test"
# os.environ["USER_AGENT"] = "vector-db-test-project"


def query():
    user_input = input("""Enter the query : \t""")
    ret = langchain_chroma.as_retriever()
    result1 = ret.invoke(user_input)
    result: list[tuple[Document, float]] = langchain_chroma.similarity_search_with_relevance_scores(user_input, 10)
    for doc in result:
        print(doc[0].metadata["file_name"], doc[1])


def delete_all():
    print("deleting the collection...")
    client.delete_collection(COLLECTION_NAME)
    print("Collection deleted")


def embedd_all():
    print("Start embedding ...")
    references = Path("references")
    all_docs=[]
    for file in references.iterdir():
        if file.is_file() and file.suffix.endswith(".txt"):
            loader = TextLoader(file)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            docs: list[Document] = text_splitter.split_documents(documents)

            for doc in docs:
                doc.metadata = {
                    "file_name": file.name,
                }

            all_docs += docs
            langchain_chroma.add_documents(documents=all_docs)

    print("Completed Embedding")


if __name__ == '__main__':
    client = chromadb.HttpClient(host="localhost", port="8005")
    langchain_chroma = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceBgeEmbeddings(),
    )
    while True:
        user_input = input("""
Enter a number (or -1 to exit):
1. To query a question
2. To embdedd a document
3. To delete the collection""")

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
            else:
                print("Invalid input. Please enter a number between 1 and 3, or -1 to exit.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    pass
