from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import torch
import ipdb
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline,AutoConfig
from langchain.document_loaders import DirectoryLoader,JSONLoader,TextLoader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import RetrievalQA,LLMChain
from langchain import PromptTemplate
from textwrap import fill
from googletrans import Translator




MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GPTQ"
# MODEL_NAME = "yentinglin/Taiwan-LLM-7B-v2.0.1-chat"
# MODEL_NAME = "yentinglin/Taiwan-LLM-13B-v2.0-chat"
# MODEL_NAME = "audreyt/Taiwan-LLM-7B-v2.1-chat-GGUF"
translator = Translator()
def translate_text(text,  target_language='zh-TW'):
    translation = translator.translate(text, dest=target_language)
    return translation.text

hf_auth="hf_key"

def loadModel(temperature):
    model_config=AutoConfig.from_pretrained(
        MODEL_NAME,
        use_auth_token=hf_auth
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",use_auth_token=hf_auth
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,token=hf_auth)

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    # create llm
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": temperature})
    return llm


def loadData(path="../../data"):
    # new load data
    loader = DirectoryLoader(path,glob="*.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    print("load data:",len(texts))

        # split add embed_query data
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    return db,embeddings
def searchData(db,text):

    # query_result = embeddings.embed_query(texts[0].page_content)
    results = db.similarity_search(text, k=2)
    return results[0].page_content
def ask(llm,db,role,text):
    template = """
    <s>[INST] <<SYS>>
    You are a travel expert and .Answer the question.
    <</SYS>>

    {context}

    {question} [/INST]
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    # prompt = PromptTemplate(template=template)

    # chain=LLMChain(llm=llm,prompt=prompt)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    result =chain(text)
    # print("all:",result)
    return result["result"]


llm=loadModel(0.9)
db,embeddings=loadData()


app = FastAPI()

# Templates configuration
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_text(request: Request, text: str = Form(...)):
    # print(text)
    text=translate_text(text,target_language="en")
    print(text)
    ans=ask(llm,db,"",text)
    print(ans)
    # ans=
    # print(ans)


    return templates.TemplateResponse("index.html", {"request": request, "processed_text":ans+"<br><br><hr><br>"+translate_text(ans)})
