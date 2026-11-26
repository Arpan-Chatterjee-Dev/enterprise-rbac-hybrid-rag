from document_ingestor import DocumentIngestor
if __name__=="__main__":
 ingestor=DocumentIngestor()
 user_input=str(input("Enter 1 for Ingestion.\nEnter 2 for Search\n"))
 match user_input:
  case "1":
   ingestor.ingest_legal_docs()
  case "2":
   user_name="adv.rao"
#    user_action=input("What you want to search")
   user_query="What is the Gifts and Hospitality rules?"
   ingestor.hybrid_search(user_name,user_query) 
  
 
 

