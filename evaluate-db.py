import asyncio
import json
from pathlib import Path

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_recall,
    context_precision, answer_correctness, answer_relevancy, faithfulness, answer_similarity,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def evaluate_dataset():
    file = Path("sample.json")
    json_data = json.loads(file.read_text())
    data = Dataset.from_dict(json_data)

    result = evaluate(
        data,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
            answer_similarity
        ],
        llm=llm,
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings())
    )

    df = result.to_pandas()
    csv = Path("result.csv")
    csv.touch()
    df.to_csv(csv)


asyncio.run(evaluate_dataset())
