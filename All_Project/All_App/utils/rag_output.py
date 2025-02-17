from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.schema import Document
from typing import List
from pydantic import Field
from All_App.llama_models import Models

# Initialize models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama
llm.temperature = 0.2

# Elasticsearch Configuration
es = Elasticsearch("http://localhost:9200")
INDEX_DOCUMENTS = "documents"

# Define Custom Elasticsearch Retriever
class ElasticsearchRetriever(BaseRetriever):
    embeddings_model: any = Field(...)
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.6)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embeddings_model.embed_query(query)

        # Retrieve documents from Elasticsearch
        query_body = {"size": 1000, "query": {"match_all": {}}}
        response = es.search(index=INDEX_DOCUMENTS, body=query_body)
        documents = response["hits"]["hits"]
        print(f"Total documents in Elasticsearch: {len(documents)}")

        if not documents:
            return []

        similarities = []
        for doc in documents:
            source = doc["_source"]
       

            if "embedding" not in source or "content" not in source:
                print("Skipping document due to missing embedding or content")
                continue  # If document has no embedding or content, skip it

            stored_embedding = np.array(source["embedding"])
            similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
           

            if similarity >= self.similarity_threshold:
                similarities.append((source, similarity))

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        relevant_docs = [
            Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            for doc, _ in similarities[:self.top_k]
        ]
        
        print(f"Final Relevant Documents Found: {len(relevant_docs)}")
        return relevant_docs


# Create retriever instance
retriever = ElasticsearchRetriever(embeddings_model=embeddings, top_k=10)

# MultiQueryRetriever using the retriever
multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions using the given context."),
    ("human", "Use the user question {input} to answer using only the {context}.")
])

# Define document processing chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
