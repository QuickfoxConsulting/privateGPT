import psycopg2
import os

# Database connection details from settings.yaml
db_config = {
    "host": "localhost",
    "port": 5434,
    "database": "quickref",
    "user": "admin",
    "password": "admin"
}

query = 'QuickXtract'

def check_db(host):
    print(f"Connecting to database at {host}:{db_config['port']}...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        cur = conn.cursor()
        
        # Check Entities
        cur.execute("SELECT id, name, type FROM private_gpt.entities WHERE name ILIKE %s", (f"%{query}%",))
        entities = cur.fetchall()
        print(f"ENTITIES MATCHING '{query}': {len(entities)}")
        for e in entities:
            print(f"  ID: {e[0]}, Name: {e[1]}, Type: {e[2]}")
            
        # Check NodeEntity Links
        cur.execute("""
            SELECT ne.node_id, ne.doc_id, e.name 
            FROM private_gpt.node_entities ne 
            JOIN private_gpt.entities e ON ne.entity_id = e.id 
            WHERE e.name ILIKE %s
        """, (f"%{query}%",))
        links = cur.fetchall()
        print(f"NODE_ENTITY LINKS MATCHING '{query}': {len(links)}")
        for l in links:
             print(f"  NodeID: {l[0]}, DocID: {l[1]}, EntityName: {l[2]}")
             
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error connecting to {host}: {e}")
        return False

# Try localhost
check_db("localhost")
