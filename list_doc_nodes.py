from private_gpt.di import global_injector
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
import sys

def list_doc_nodes(filename):
    node_store = global_injector.get(NodeStoreComponent)
    docstore = node_store.doc_store
    
    found = []
    for node_id, node in docstore.docs.items():
        node_meta = node.metadata or {}
        n_file = node_meta.get("file_name")
        
        if n_file == filename:
            found.append((node_id, node_meta.get("page"), node.get_content()[:50].replace("\n", " "), node_meta.get("document_id")))

    print(f"Found {len(found)} nodes for file {filename}")
    for nid, page, text, doc_id in sorted(found, key=lambda x: (x[1] or 0, x[0])):
        print(f"ID: {nid} | DocID: {doc_id} | Page: {page} | Text: {text}...")

if __name__ == "__main__":
    filename = "Citizen_Rtgs_aa-3.pdf"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    list_doc_nodes(filename)
