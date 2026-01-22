import logging
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.tool import ToolDefinition, ToolCategory

# Integrations
from private_gpt.server.tools.integrations.gmail_tool import GmailTool
from private_gpt.server.tools.integrations.calendar_tool import CalendarTool
from private_gpt.server.tools.integrations.sheets_tool import SheetsTool
from private_gpt.server.tools.integrations.docs_tool import DocsTool
from private_gpt.server.tools.integrations.crawl4ai_tool import Crawl4AITool
from private_gpt.server.tools.integrations.serper_tool import SerperTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_tools():
    db = SessionLocal()
    try:
        # 1. Ensure Categories Exist
        categories = {
            "integration": {"desc": "External service integrations", "icon": "🔌"},
            "web": {"desc": "Web browsing and search capabilities", "icon": "🌐"},
        }
        
        category_map = {}
        for name, data in categories.items():
            cat = db.query(ToolCategory).filter_by(name=name).first()
            if not cat:
                cat = ToolCategory(
                    name=name,
                    description=data["desc"],
                    icon=data["icon"]
                )
                db.add(cat)
                db.commit()
                db.refresh(cat)
                logger.info(f"Created category: {name}")
            category_map[name] = cat

        # 2. Define Tools to Seed
        # Format: (Class, Name, Display Name, Category Name, Icon, ValidConfigKeys)
        tools_to_seed = [
            (GmailTool, "gmail_tool", "Gmail Integration", "integration", "📧"),
            (CalendarTool, "calendar_tool", "Google Calendar", "integration", "📅"),
            (SheetsTool, "sheets_tool", "Google Sheets", "integration", "📊"),
            (DocsTool, "docs_tool", "Google Docs", "integration", "📝"),
            (Crawl4AITool, "crawl4ai_scraper", "Web Crawler", "web", "🕷️"),
            (SerperTool, "serper_tool", "Serper Search", "web", "🔍"),
        ]

        for ToolClass, tool_name, display_name, cat_name, icon in tools_to_seed:
            try:
                # Instantiate dummy tool to get metadata/schema
                # We pass minimal dummy args for initialization
                # NOTE: Some tools might require args in init, check signature
                # BaseMCPTool accepts matches: def __init__(self, user_id: int = 0, config: Dict[str, Any] = None):
                # But Crawl4AITool has: def __init__(self, verbose: bool = False, *args, **kwargs):
                
                # We'll use kwargs catch-all or specific instantiations if needed
                if ToolClass == Crawl4AITool:
                    instance = ToolClass(verbose=False, user_id=0, config={})
                else:
                    instance = ToolClass(user_id=0, config={})
                
                metadata = instance.metadata  # LlamaIndex ToolMetadata
                config_schema = ToolClass.get_config_schema()
                module_path = ToolClass.__module__
                class_name = ToolClass.__name__
                
                # Check existance
                tool_def = db.query(ToolDefinition).filter_by(name=tool_name).first()
                if not tool_def:
                    tool_def = ToolDefinition(
                        name=tool_name,
                        display_name=display_name,
                        category_id=category_map[cat_name].id,
                        version="1.0.0",
                        module_path=module_path,
                        class_name=class_name,
                        description=metadata.description,
                        icon=icon,
                        config_schema=config_schema,
                        is_official=True,
                        is_verified=True
                    )
                    db.add(tool_def)
                    db.commit()
                    logger.info(f"Successfully registered {display_name}")
                else:
                    # Update existing definition if needed (e.g. description or schema changed)
                    tool_def.description = metadata.description
                    tool_def.config_schema = config_schema
                    tool_def.module_path = module_path
                    tool_def.class_name = class_name
                    db.commit()
                    logger.info(f"Updated {display_name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {tool_name}: {e}")
                db.rollback()

    except Exception as e:
        logger.error(f"Error seeding tools: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_tools()
