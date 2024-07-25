import os
import asyncio
import json
from pathlib import Path

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    context_recall,
    context_precision,
)
from langchain_openai import OpenAI
from ragas import evaluate

llm = OpenAI(model_name="gpt-3.5-turbo")





async def evaluate_dataset():
    file = Path("sample.json")
    json_data = json.loads(file.read_text())
    data = Dataset.from_dict(json_data)

    result = evaluate(
        data,
        metrics=[
            context_precision,
            context_recall,
        ],
        # llm=LangchainLLMWrapper(get_llm("llama3")),
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings())
    )

    df = result.to_pandas()
    csv = Path("result.csv")
    csv.touch()
    df.to_csv(csv)


asyncio.run(evaluate_dataset())
