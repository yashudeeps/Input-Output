import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_MODEL")

def embedding():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        openai_api_version=azure_api_version,
    )
    return embeddings

def chat():
    chat_model = AzureChatOpenAI(
        openai_api_version=azure_api_version,
        azure_deployment=azure_chat_deployment,
        temperature=0.8
    )
    return chat_model
