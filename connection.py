import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

vectorizer = os.getenv("VECTORIZER_MODEL")
LLM = os.getenv("LLM")
cluster_url = os.getenv("CLUSTER_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
huggingface_api = os.getenv("HuggingFace_API")
openrouter_key = os.getenv("OPENROUTER_KEY")
base_url = os.getenv("BASE_URL")


def connect_db():

    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers={
                "X-HuggingFace-Api-Key": huggingface_api,
            },
        )
        print(
            f"Connection Status:{client.is_ready()} Connected Successfully\nWarning! Remember to close connection using 'client.close()'"
        )
    except Exception as e:
        print(e)
    return client


def huggingFace_vectorizer(model=vectorizer):
    return wc.Configure.Vectorizer.text2vec_huggingface(model=model)


def LLM_pipeline(prompt, base_url=base_url, key=openrouter_key, LLM=LLM):
    client = OpenAI(base_url=base_url, api_key=openrouter_key)
    completion = client.chat.completions.create(
        model=LLM,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Assistant that Answers User Queries.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content
