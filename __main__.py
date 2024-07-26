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
    question = [
        "How we can prevent cyber attack?",
        "How AI is being used in field of agriculture?",
        "How machine learning used to detect cyber attack?",
        "How geopolitic risk analysis affect market volatility?",
        "How sensors along with ML used to improve agriculture?",
        "How market behaves during a pendamic?",
        "What is MLOps and how it is used to automate machine learning tasks?",
        "What is MLOps lifecycle?",
        "What is the architecture of MLOps?",
        "What are the model registries are used for MLOps?",
        "What are the challenges in MLOps?",
        "How does MLOps streamline machine learning deployment?",
        "What are the best practices for implementing MLOps?",
        "How does continuous integration work in MLOps?",
        "What tools are commonly used in MLOps pipelines?",
        "How does MLOps handle versioning of ML models?",
        "What are the key components of an MLOps framework?",
        "How is monitoring implemented in MLOps?",
        "What are the benefits of using MLOps for model scalability?",
        "How does MLOps ensure reproducibility in ML experiments?",
        "What role does data management play in MLOps?",
        "How does MLOps integrate with DevOps practices?",
        "What are the security considerations in MLOps?",
        "How can MLOps improve model performance in production?",
        "What is the role of feature stores in MLOps?",
        "How does MLOps support A/B testing of ML models?",
        "What are the key metrics to monitor in MLOps?",
        "How does MLOps handle model drift?",
        "What are the common challenges faced during MLOps implementation?",
        "How can MLOps help in reducing time to market for ML models?",
        "What is the significance of orchestration in MLOps?",
        "How does MLOps facilitate collaboration between data scientists and engineers?",
        "What are the common pitfalls to avoid in MLOps?",
        "How does MLOps handle multi-cloud deployments?",
        "What are the different deployment strategies in MLOps?",
        "How does MLOps ensure compliance with data regulations?",
        "What is the role of automated testing in MLOps?",
        "How can MLOps improve the efficiency of ML pipelines?",
        "What are the benefits of using containers in MLOps?",
        "How does MLOps handle large-scale data processing?",
        "What are the challenges in securing ML models in MLOps?",
        "How does MLOps support continuous learning?",
        "What are the different stages of the MLOps lifecycle?",
        "How does MLOps manage data quality issues?",
        "What are the best practices for scaling MLOps?",
        "How does MLOps handle data privacy concerns?",
        "What are the benefits of using Kubernetes in MLOps?",
        "How can MLOps improve the reliability of ML systems?",
        "What are the key considerations for building an MLOps team?",
        "How does MLOps integrate with CI/CD pipelines?",
        "What is the role of automation in MLOps?",
        "How does MLOps handle model interpretability?",
        "What are the benefits of using MLflow in MLOps?",
        "How does MLOps support model governance?",
        "What are the challenges of implementing MLOps in a large organization?",
        "How can MLOps help in managing technical debt?",
        "What are the different types of cyberattacks that can be detected using machine learning?",
        "How does anomaly detection work in cybersecurity?",
        "What are the common machine learning algorithms used in cybersecurity?",
        "How can machine learning help in identifying phishing attacks?",
        "What is the role of machine learning in intrusion detection systems?",
        "How can supervised learning be used to improve cybersecurity?",
        "What are the benefits of using unsupervised learning in detecting cyber threats?",
        "How can machine learning help in predicting cyberattacks?",
        "What are the challenges of using machine learning in cybersecurity?",
        "How can reinforcement learning be applied to cybersecurity?",
        "What is the role of feature engineering in cybersecurity models?",
        "How can machine learning improve malware detection?",
        "What are the ethical considerations of using AI in cybersecurity?",
        "How can machine learning help in network security?",
        "What are the limitations of current machine learning models in cybersecurity?",
        "How can machine learning be used to detect insider threats?",
        "What are the best practices for training machine learning models for cybersecurity?",
        "How can machine learning help in the response to cyber incidents?",
        "What are the key features used in machine learning models for cybersecurity?",
        "How can machine learning help in reducing false positives in threat detection?",
        "What is the impact of AI on the future of cybersecurity?",
        "How can machine learning be used to protect IoT devices from cyberattacks?",
        "What are the common datasets used for training machine learning models in cybersecurity?",
        "How can machine learning help in identifying vulnerabilities in software?",
        "What are the benefits of using machine learning for endpoint security?",
        "How can machine learning help in securing cloud environments?",
        "What are the challenges of deploying machine learning models in cybersecurity?",
        "How can machine learning help in improving threat intelligence?",
        "What are the different types of cyber threats that machine learning can detect?",
        "How can machine learning help in automated threat hunting?",
        "What are the best practices for evaluating machine learning models in cybersecurity?",
        "How can machine learning help in protecting against ransomware attacks?",
        "What are the challenges of using machine learning in real-time threat detection?",
        "How can machine learning help in identifying advanced persistent threats?",
        "What are the different types of cyberattacks that can be prevented using machine learning?",
        "How can machine learning help in reducing the impact of cyberattacks?",
        "What are the benefits of using machine learning for security analytics?",
        "How can machine learning help in protecting critical infrastructure?",
        "What are the common techniques used in machine learning for cybersecurity?",
        "How can machine learning help in improving the accuracy of threat detection?",
        "What are the different types of machine learning models used in cybersecurity?",
        "How can machine learning help in enhancing incident response capabilities?",
        "What are the challenges of implementing machine learning in agriculture?",
        "How can machine learning help in livestock management?",
        "What are the applications of ML in weather forecasting for agriculture?",
        "How can machine learning improve supply chain management in agriculture?",
        "What are the ethical considerations of using AI in agriculture?",
        "How can machine learning help in reducing food waste?",
        "What are the benefits of using ML for crop disease detection?",
        "How can machine learning help in precision fertilization?",
        "What are the applications of ML in agricultural market analysis?",
        "How can machine learning help in improving farm productivity?",
        "What are the challenges of using ML in remote sensing for agriculture?",
        "How can AI help in sustainable farming practices?",
        "What are the benefits of using machine learning in smart greenhouses?",
        "How can machine learning help in optimizing crop rotation schedules?",
        "What are the applications of AI in vertical farming?",
        "How can machine learning help in reducing the use of pesticides?",
        "What are the benefits of using ML for agricultural policy planning?",
        "How can AI assist in agricultural research and development?",
        "What are the challenges of integrating AI with traditional farming methods?",
        "How can machine learning help in managing agricultural risks?",
        "What are the applications of ML in aquaculture?",
        "How can machine learning improve the efficiency of food distribution?",
        "What are the benefits of using AI for farm management systems?",
        "How can machine learning help in the development of new crop varieties?",
        "What are the challenges of using ML for agricultural data analysis?",
        "How can AI help in the adoption of regenerative agriculture?",
        "What are the benefits of using machine learning in agroforestry?",
        "How can AI assist in climate-smart agriculture?",
        "What are the applications of ML in plant breeding?",
        "How can machine learning help in reducing greenhouse gas emissions in agriculture?",
        "What are the challenges of using AI for crop phenotyping?",
        "How can machine learning help in improving food security?",
        "What are the benefits of using ML for water resource management in agriculture?",
        "How can AI assist in agricultural extension services?",
        "What are the applications of machine learning in agricultural finance?",
        "How can machine learning help in optimizing harvest schedules?",
        "What are the benefits of using AI for post-harvest management?",
        "How can machine learning help in reducing the environmental impact of agriculture?",
        "What are the challenges of using AI for agricultural innovation?",
        "How can machine learning help in enhancing biodiversity on farms?",
        "What are the benefits of using AI for agricultural marketing?",
        "How can machine learning help in improving farm labor management?",
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
    # get_chromadb_client().delete_collection(COLLECTION_NAME)
    get_qdrant_client().delete_collection(COLLECTION_NAME)
    print("Collection deleted")


def embedd_all():
    print("Start embedding ...")
    references = Path("references")
    all_docs = []
    # get_qdrant_client().create_collection(collection_name=COLLECTION_NAME,
    #                                       vectors_config=models.VectorParams(distance=models.Distance.COSINE))
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

    # vectorstore = Chroma(
    #     client=get_chromadb_client(),
    #     collection_name=COLLECTION_NAME,
    #     embedding_function=HuggingFaceBgeEmbeddings(),
    # )

    vectorstore = Qdrant(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embeddings=HuggingFaceBgeEmbeddings()
    )

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
