import logging
import sys
from pathlib import Path
from llama_index.core.storage import StorageContext
from private_gpt.di import global_injector
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

def check_node(node_id):
    node_store = global_injector.get(NodeStoreComponent)
    docstore = node_store.doc_store
    
    print(f"Checking node_id: {node_id}")
    if node_id in docstore.docs:
        print(f"FOUND in docstore!")
        node = docstore.get_node(node_id)
        print(f"Text: {node.get_content()[:100]}...")
        print(f"Metadata: {node.metadata}")
    else:
        print(f"NOT FOUND in docstore.")
        print(f"Total nodes in docstore: {len(docstore.docs)}")

if __name__ == "__main__":
    node_id = "425ea3a1-2280-42a4-87a6-a957ee72a6e3"
    if len(sys.argv) > 1:
        node_id = sys.argv[1]
    check_node(node_id)
