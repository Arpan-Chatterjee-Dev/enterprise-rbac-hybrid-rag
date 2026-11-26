print("starting", flush=True)
from dotenv import load_dotenv
from langchain_qdrant import FastEmbedSparse,QdrantVectorStore, RetrievalMode
from pandas import Series
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance,VectorParams,SparseVectorParams,Modifier
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.datamodel.pipeline_options import PdfPipelineOptions,HeadingHierarchyOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter,PdfFormatOption
from langchain_core.documents import Document
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from transformers import AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from qdrant_client.http.models import Field,Filter,FieldCondition,Match,MatchValue,MatchAny
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
class DocumentIngestor:

  load_dotenv()
  def doc_ingestion_setup(self):
    
    self.customized_chunk_texts:list[Document]=[]
    self.pdfpipelines=PdfPipelineOptions(allow_external_plugins=True,heading_hierarchy_options=HeadingHierarchyOptions(enabled=True))
    self.tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.EMBED_MODEL)
    self.chunker= HybridChunker(tokenizer=self.tokenizer,merge_peers=True,max_tokens=256)
    self.pdf_converter=DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=self.pdfpipelines)})
      


  def __init__(self):
   self.EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   self.collection_name="veridian_legal"
   self.SPARSE_VECTOR_NAME="veridion_legal_sparse"
   self.DOC_PATH="./veridian_legal_dataset/"
   self.client=QdrantClient(path="/local/vectorDb")
   self.dense_embedder=HuggingFaceEmbeddings(model_name = self.EMBED_MODEL,
             model_kwargs = {"device": "cuda"},
             encode_kwargs = {"normalize_embeddings": True})
   self.sparse_embedder=FastEmbedSparse(model_name="Qdrant/bm25",batch_size=32)
   self.vector_store:QdrantVectorStore=QdrantVectorStore(client=self.client,collection_name=self.collection_name,embedding=self.dense_embedder,metadata_payload_key=None,
                                        sparse_embedding=self.sparse_embedder,retrieval_mode=RetrievalMode.HYBRID,sparse_vector_name=self.SPARSE_VECTOR_NAME)
         
      

  def get_chunk_type(self,doc_items) -> str:
    labels= {str(item["label"]).lower() for item in doc_items}
    if "table" in labels:
     return "table" 
    if "code" in labels:
     return "code" 
    heading_labels = {"section_header", "title"}
    if labels and labels <= heading_labels:
     return "heading"
    return "text"
  
  
  def read_legal_dataset(self,data:Series,filepath:str):
    
      if filepath.lower().endswith(".pdf"):
       chunked_docs=DoclingLoader(file_path=filepath,converter=self.pdf_converter,chunker=self.chunker).load()
      elif filepath.lower().endswith(".md"):
       chunked_docs=DoclingLoader(file_path=filepath,chunker=self.chunker,export_type=ExportType.DOC_CHUNKS).load()
      else:
        raise ValueError(f"Unsupported file type: {filepath}")
      for doc in chunked_docs:
        
        heading_list=doc.metadata["dl_meta"]["headings"]
        heading=">".join(heading_list) if heading_list else "" 
        content=doc.page_content
        self.customized_chunk_texts.append(Document(page_content=f"{heading}\n\n{content}",
                                                metadata={"source_document": data["source_document"],
                                                          "section_title":heading,
                                                          "original_text":content,
                                                          "access_roles":data["access_roles"],
                                                          "collection":data["collection"],
                                                          "matter_id":data["matter_id"],
                                                          "doc_id":data["doc_id"],
                                                          "chunk_type":self.get_chunk_type(doc.metadata["dl_meta"]["doc_items"])}))
        

  def create_embedding(self,customized_chunk_texts):
  
   if self.client.collection_exists(collection_name=self.collection_name):
     self.client.delete_collection(collection_name=self.collection_name)
   if not self.client.collection_exists(collection_name=self.collection_name):
    self.client.create_collection(collection_name=self.collection_name,
                              sparse_vectors_config={self.SPARSE_VECTOR_NAME:SparseVectorParams(modifier=Modifier.IDF)},
                              vectors_config=VectorParams(distance=Distance.COSINE,size=len(self.dense_embedder.embed_query("test"))))
   for field in ["collection","matter_id","access_roles"]:
    self.client.create_payload_index(collection_name=self.collection_name,field_name=field,field_schema="keyword")
   ids=self.vector_store.add_documents(customized_chunk_texts)
   print(f"Upsert Done {len(ids)}")

  def hybrid_search(self,user_name:str,seach_query:str):
    df=pd.read_json(self.DOC_PATH+"users_seed.json")
    for _,item in df.iterrows():
      if item["username"]==user_name:
        current_user_roles=[item["role"]]
        current_user_matter=item["matters"]
        conditions = [
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=current_user_roles)
            )
        ]

        if "*" not in current_user_matter:
            conditions.append(
                FieldCondition(
                    key="matter_id",
                    match=MatchAny(any=["NONE", *current_user_matter])
                )
            )
        qdrant_filter = Filter(must=conditions)
        retrievar=self.vector_store.as_retriever(search_kwargs={"k":10,"filter":qdrant_filter})
        results=retrievar.invoke(seach_query)
        print(results)

  def ingest_legal_docs(self):  
    try:
        self.doc_ingestion_setup()
        manifest=pd.read_json(self.DOC_PATH+"manifest.json")
        for _,item in manifest.iterrows():
         if item["path"]:    
          self.read_legal_dataset(item,self.DOC_PATH+item["path"])
        self.create_embedding(self.customized_chunk_texts)
    except Exception as e:
        print("Error occurred during document ingestion",e)
    finally:
        self.client.close()

