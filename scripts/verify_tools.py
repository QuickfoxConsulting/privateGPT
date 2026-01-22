from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.tool import ToolDefinition

def verify_tools():
    db = SessionLocal()
    try:
        tools = db.query(ToolDefinition).all()
        print(f"Total tools found: {len(tools)}")
        print("-" * 50)
        for tool in tools:
            print(f"ID: {tool.id} | Name: {tool.name} | Category: {tool.category.name} | Verified: {tool.is_verified}")
    finally:
        db.close()

if __name__ == "__main__":
    verify_tools()
