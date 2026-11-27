print("starting ingestion", flush=True)
from pathlib import Path

from dotenv import load_dotenv
from langchain_qdrant import RetrievalMode
from pandas import Series
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
import pandas as pd

class DocumentIngestor:

 EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
 collection_name="veridian_legal"
 SPARSE_VECTOR_NAME="veridian_legal_sparse"
 BASE_DIR = Path(__file__).resolve().parent
 PROJECT_DIR = BASE_DIR.parent
 DOC_PATH = PROJECT_DIR / "veridian_legal_dataset"


 load_dotenv()

 def __init__(self,client,dense_embedder,vector_store):
    self.client=client
    self.vector_store=vector_store    
    self.customized_chunk_texts:list[Document]=[]
    self.pdfpipelines=PdfPipelineOptions(allow_external_plugins=True,heading_hierarchy_options=HeadingHierarchyOptions(enabled=True))
    self.tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.EMBED_MODEL)
    self.chunker= HybridChunker(tokenizer=self.tokenizer,merge_peers=True,max_tokens=256)
    self.pdf_converter=DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=self.pdfpipelines)})
    self.dense_embedder=dense_embedder
   
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
        

 def create_embedding(self,customized_chunk_texts)->int:
   for field in ["metadata.collection","metadata.matter_id","metadata.access_roles"]:
    self.client.create_payload_index(collection_name=self.collection_name,field_name=field,field_schema="keyword")
   ids=self.vector_store.add_documents(customized_chunk_texts)
   return len(ids)

 def ingest_legal_docs(self)->int:  
    try:
        manifest=pd.read_json(self.DOC_PATH / "manifest.json")
        for _,item in manifest.iterrows():
         if item["path"]:    
          self.read_legal_dataset(item,str(self.DOC_PATH / item["path"]))
        return self.create_embedding(self.customized_chunk_texts)
    except Exception as e:
        print("Error occurred during document ingestion",e)