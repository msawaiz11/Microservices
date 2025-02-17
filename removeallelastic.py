from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
es.delete_by_query(index="documents", body={"query": {"match_all": {}}})
es.delete_by_query(index="processed_files", body={"query": {"match_all": {}}})
