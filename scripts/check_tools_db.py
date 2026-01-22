from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.tool import UserToolInstallation, ToolDefinition, ToolCategory
from private_gpt.users.models.user import User
from sqlalchemy import select

def check_db():
    db = SessionLocal()
    try:
        print("--- Users ---")
        users = db.execute(select(User)).scalars().all()
        for u in users:
            print(f"User ID: {u.id}, Email: {u.email}")
            
        print("\n--- Tool Definitions ---")
        tools = db.execute(select(ToolDefinition)).scalars().all()
        for t in tools:
            print(f"Tool ID: {t.id}, Name: {t.name}, Display: {t.display_name}")
            
        print("\n--- User Tool Installations ---")
        installs = db.execute(select(UserToolInstallation)).scalars().all()
        if not installs:
            print("No installations found!")
        for i in installs:
            print(f"ID: {i.id}, User ID: {i.user_id}, Tool ID: {i.tool_id}, Enabled: {i.is_enabled}, Created At: {i.installed_at}")
            # config = i.config
            # print(f"  Config: {config}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_db()
