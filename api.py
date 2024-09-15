from modules import QARetriever

from typing import Union
from fastapi import FastAPI

app = FastAPI()
bot = QARetriever(url="https://textnet.kr/about")


@app.get("/")
def read_root():
    return {"message": "Hello World: Successfully connedted to FastAPI"}


@app.get("/api")
def chat(query: Union[str, None] = None):
    if query is None:
        return {"message": "Error: No query received"}
    res = bot(query)
    return res
