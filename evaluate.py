import openai
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
import asyncio
import pandas as pd 
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
import streamlit as st 
from src import ingest_pipeline, index_builder
import os

def setup_openai(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2):
    openai.api_key = api_key
    Settings.llm = OpenAI(model=model, temperature=temperature)

def create_document_and_splitter(text, chunk_size=20, chunk_overlap=5, seperator=" "):
    doc = Document(text=text)
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seperator=seperator
    )
    nodes = splitter.get_nodes_from_documents([doc])
    return nodes

def create_vector_store_index(nodes):
    vector_index = VectorStoreIndex(nodes)
    query_engine = vector_index.as_query_engine()
    return query_engine

def generate_questions(nodes, num_questions_per_chunk=1):
    dataset_generator = RagDatasetGenerator(nodes, num_quesiton_per_chunk=num_questions_per_chunk)
    eval_questions = dataset_generator.generate_questions_from_nodes()
    return eval_questions.to_pandas()

async def evaluate_async(query_engine, df):
    correctness_evaluator = CorrectnessEvaluator()
    faithfulness_evaluator = FaithfulnessEvaluator()
    relevancy_evaluator = RelevancyEvaluator()

    runner = BatchEvalRunner(
        {
            "correctness": correctness_evaluator,
            "faithfulness": faithfulness_evaluator,
            "relevancy": relevancy_evaluator
        },
        show_progress=True
    )

    eval_result = await runner.aevaluate_queries(
        query_engine=query_engine,
        queries=[question for question in df['query']],
    )

    return eval_result

def aggregate_results(df, eval_result):
    data = []
    for i, question in enumerate(df['query']):
        correctness_result = eval_result['correctness'][i]
        faithfulness_result = eval_result['faithfulness'][i]
        relevancy_result = eval_result['relevancy'][i]
        data.append({
            'Query': question,
            'Correctness respones': correctness_result.respones,
            'Correctness passing': correctness_result.passing,
            'Correctness feedback': correctness_result.feedback,
            'Correctness score': correctness_result.score,
            'Faithfulness respones': faithfulness_result.respones,
            'Faithfulness passing': faithfulness_result.passing,
            'Faithfulness feedback': faithfulness_result.feedback,
            'Faithfulness score': faithfulness_result.score,
            'Relevancy respones': relevancy_result.respones,
            'Relevancy passing': relevancy_result.passing,
            'Relevancy feedback': relevancy_result.feedback,
            'Relevancy score': relevancy_result.score,
        })

    df_result = pd.DataFrame(data)
    return df_result

def print_average_scores(df):
    correctness_scores = df['Correctness score'].mean()
    faithfulness_scores = df['Faithfulness score'].mean()
    relevancy_scores = df['Relevancy score'].mean()
    print(f"Correctness scores: {correctness_scores}")
    print(f"Faithfulness scores: {faithfulness_scores}")
    print(f"Relevancy scores: {relevancy_scores}")

def main():
    nest_asyncio.apply()

    api_key = st.secrets.openai.OPENAI_API_KEY
    setup_openai(api_key=api_key)

    nodes = ingest_pipeline.ingest_documents()

    index = index_builder.build_indexes(nodes)
    dsm5_engine = index.as_query_engine(
        similarity_top_k=3,
    )

    df = generate_questions(nodes)

    eval_result = asyncio.run(evaluate_async(query_engine=dsm5_engine, df=df))
    df_result = aggregate_results(df, eval_result)

    correctness_scores, faithfulness_scores, relevancy_scores = print_average_scores(df_result)

    os.makedirs("eval_results", exist_ok=True)
    df_result.to_csv("eval_results/evaluation_results.csv", index=False)
    df.to_csv("eval_results/evaluation_questions.csv", index=False)
    with open("eval_results/average_scores.txt", "w") as f:
        f.write(f"Correctness scores: {correctness_scores}\n")
        f.write(f"Faithfulness scores: {faithfulness_scores}\n")
        f.write(f"Relevancy scores: {relevancy_scores}\n")