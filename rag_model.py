from langchain_chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from azure_models import embedding, chat

embeddings = embedding()
chat_model = chat()

class rag(BaseModel):
    Response: str = Field(description="Respond to the Query according to the Data.")

vector_db = None
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})
structured_llm = chat_model.with_structured_output(rag)
chat_history = ""
query = input("\nUSER:")

while query.lower() not in ["exit","exit."] :
    res = retriever.invoke(query)
    print(res[0].metadata)
    prompt = "Understand the given chat history and data," + chat_history + res[0].page_content + "and answer the following Query in a well formatted way: " + query + ".And don't mention about the chat history."
    assistant_response = structured_llm.invoke(prompt)
    chat_history = "\n\nUser:" + query + "\nAssistant:" + assistant_response.Response + chat_history
    
    print("ASSISTANT:", assistant_response.Response)
    print()
    
    query = input("USER:")