print("starting retrieval", flush=True)
from dotenv import load_dotenv
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from pandas import Series
from qdrant_client.http.models import Distance,VectorParams,SparseVectorParams,Modifier
from langchain_core.documents import Document
from qdrant_client.http.models import Field,Filter,FieldCondition,Match,MatchValue,MatchAny
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from dto.user import User

class DocumentRetriever:

 ALL_COLLECTIONS = ["general", "contracts", "litigation", "compliance", "ip"]

 ROLE_COLLECTIONS: dict[str, list[str]] = {
    "corporate_counsel": ["contracts", "ip", "general"],
    "litigation_counsel": ["litigation", "ip", "general"],
    "compliance_officer": ["compliance", "general"],
    "paralegal": ["contracts", "general"],
    "general_counsel": ALL_COLLECTIONS,
 }
 CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
 collection_name="veridian_legal"
 DOC_PATH="../veridian_legal_dataset/"

 load_dotenv()

 def __init__(self,client,vector_store):
  self.client=client
  self.vector_store:QdrantVectorStore=vector_store
         
 
 def hybrid_search(self,user:User,search_query:str)->list[Document]:
  
        current_user_roles=[user.role]
        current_user_matter=user.matters
        current_user_collection_access:list[str]=self.ROLE_COLLECTIONS.get(user.role,[])
        conditions = [
            FieldCondition(
                key="metadata.access_roles",
                match=MatchAny(any=current_user_roles)
            ),
            FieldCondition(key="metadata.collection",
                match=MatchAny(any=current_user_collection_access))
        ]
 
        if "*" not in current_user_matter:
            conditions.append(
                FieldCondition(
                    key="metadata.matter_id",
                    match=MatchAny(any=["NONE", *current_user_matter])
                )
            )
        qdrant_filter = Filter(must=conditions)
        broad_retrievar=self.vector_store.as_retriever(search_kwargs={"k":10,"filter":qdrant_filter})
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.CROSS_ENCODER_MODEL,model_kwargs={"device":"cuda"})
        re_ranker = CrossEncoderReranker(model=cross_encoder,top_n=3)
        re_ranking_retrievar=ContextualCompressionRetriever(base_compressor=re_ranker,base_retriever=broad_retrievar)
        results=re_ranking_retrievar.invoke(search_query)
        return results