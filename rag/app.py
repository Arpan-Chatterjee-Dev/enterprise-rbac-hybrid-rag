from contextlib import asynccontextmanager
from pathlib import Path
from ingestor import DocumentIngestor
from retriever import DocumentRetriever
from langchain_qdrant import FastEmbedSparse,QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance,VectorParams,SparseVectorParams,Modifier
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI
from langchain_core.documents import Document
import pandas as pd
from dto.user import User

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "veridian_legal_dataset"
USERS_SEED_PATH = DATASET_DIR / "users_seed.json"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
collection_name="veridian_legal"
SPARSE_VECTOR_NAME="veridian_legal_sparse"
DOC_PATH="../veridian_legal_dataset/"

def load_users()->{str,User}:
    users:dict[str,User] ={}
    df=pd.read_json(USERS_SEED_PATH)
    users={user_row["username"]:User(user_row["username"],user_row["password"],user_row["display_name"],user_row["role"],user_row["matters"]) 
                 for _,user_row in df.iterrows()}
    return users   
       
@asynccontextmanager
async def setup(app:FastAPI):
 app.state.users=load_users()
 app.state.client=QdrantClient(path="/local/vectorDb")
 app.state.dense_embedder=HuggingFaceEmbeddings(model_name = EMBED_MODEL,
                model_kwargs = {"device": "cuda"},
                encode_kwargs = {"normalize_embeddings": True})
 app.state.sparse_embedder=FastEmbedSparse(model_name="Qdrant/bm25",batch_size=32)
 if app.state.client.collection_exists(collection_name=collection_name):
        app.state.client.delete_collection(collection_name=collection_name)
 if not app.state.client.collection_exists(collection_name=collection_name):
       app.state.client.create_collection(collection_name=collection_name,
                                 sparse_vectors_config={SPARSE_VECTOR_NAME:SparseVectorParams(modifier=Modifier.IDF)},
                                 vectors_config=VectorParams(distance=Distance.COSINE,size=len(app.state.dense_embedder.embed_query("test"))))
 app.state.vector_store= QdrantVectorStore(
               client=app.state.client,
               collection_name=collection_name,
               embedding=app.state.dense_embedder,
               sparse_embedding=app.state.sparse_embedder,
               retrieval_mode=RetrievalMode.HYBRID,
               sparse_vector_name=SPARSE_VECTOR_NAME,
           )
 yield
 app.state.client.close()
app= FastAPI(lifespan=setup) 

@app.get("/login")
def login(username:str,password:str)->tuple[str,User|None]:
   if not username or not password:
    return ("Invalid username or password",None)
   else:
      user=app.state.users.get(username)
      if not user:
       return ("Invalid username",None)
      else:
       if user.password==password:
        return (f"Hi! {user.display_name}",user)
       else:
         return ("Invalid password",None)  

@app.post("/chat")
def chat(user_query:str)->str:
 user_name="adv.rao"
#  user_query="What is the Gifts and Hospitality rules?"
 retrievar=DocumentRetriever(app.state.client,app.state.vector_store)
 results:list[Document]=retrievar.hybrid_search(app.state.users[user_name],user_query) 
 if isinstance(results,list):
  return "\n\n".join(result.page_content for result in results)
 return results

@app.post("/ingest")
def ingest()->str:
    ingestor=DocumentIngestor(app.state.client,app.state.dense_embedder,app.state.vector_store)
    return f"{ingestor.ingest_legal_docs()}"

