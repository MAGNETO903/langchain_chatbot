# import
import uvicorn
from fastapi import FastAPI
from langchain.llms import LlamaCpp
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download

import time

print(time.ctime(), "app launched")




# read, parse and split documents
root_dir = './'

# PDFs
pdf_loader_1 = PyPDFLoader(root_dir+'documents/doc1.pdf')
pdf_loader_2 = PyPDFLoader(root_dir+'documents/doc2.pdf')

# csv
csv_loader = CSVLoader(
    file_path=root_dir+'documents/cards.csv',
    encoding="utf-8",
    csv_args={'delimiter': ','}
)

loader_all = MergedDataLoader(loaders=[
    csv_loader,
    pdf_loader_1,
    pdf_loader_2
])

documents = loader_all.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

texts = text_splitter.split_documents(documents)
print(time.ctime(), "documents loaded")
# creating vector DB

embedder = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
vec_db = FAISS.from_documents(texts, embedder)
print(time.ctime(), "vector DB ready")

# our LLM
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_PATH = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    max_tokens=2000,
    n_ctx = 1024*3,
    top_p=0.9, # Verbose is required to pass to the callback manager
    lang="ru",
)
print(time.ctime(), "llm ready")
# building final chain
prompt_template = """
Используйте следующие фрагменты контекста, чтобы ответить на вопрос в конце.
{context}
Вопрос: {question}
Полезный ответ:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vec_db.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)

print(time.ctime(), "chain ready")
app = FastAPI()
@app.get("/message")
def message(user_id: str, message: str):
    llm_response = qa_chain({"query": message})
    return {"message": llm_response['result']}

if __name__ == '__main__':
    uvicorn.run(app)
