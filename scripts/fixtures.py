import json
import os
import importlib
import pytest
from sqlalchemy.orm import Session
from sqlalchemy import inspect

def load_fixture(session: Session, model_name: str, fixture_path: str):
    """
    Load fixtures from a JSON file into the database only if they do not already exist.

    Args:
        session: SQLAlchemy session
        model_name: Fully qualified model name (e.g., 'private_gpt.models.User')
        fixture_path: Path to the JSON file with fixture data
    """
    try:
        module_path, class_name = model_name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        with open(fixture_path, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

        # Get primary key column(s)
        pk_keys = [key.name for key in inspect(model_class).primary_key]

        inserted = []
        for fixture in fixtures:
            # Build primary key filter
            pk_filter = {key: fixture[key] for key in pk_keys if key in fixture}
            if not pk_filter:
                continue  # Skip if fixture doesn't include primary key(s)

            # Check for existence
            existing = session.query(model_class).filter_by(**pk_filter).first()
            if existing:
                continue

            instance = model_class(**fixture)
            session.add(instance)
            inserted.append(instance)

        if inserted:
            session.commit()

        return inserted
    except Exception as e:
        print(f"Error loading fixture {fixture_path}: {e}")
        session.rollback()
        return []


def load_all_fixtures(db_session):
    """Load all fixtures."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    
    companies = load_fixture(db_session, 'private_gpt.users.models.Company', 
                           os.path.join(fixtures_dir, 'companies.json'))
    departments = load_fixture(db_session, 'private_gpt.users.models.Department', 
                       os.path.join(fixtures_dir, 'departments.json'))
    users = load_fixture(db_session, 'private_gpt.users.models.User', 
                       os.path.join(fixtures_dir, 'users.json'))
    
    roles = load_fixture(db_session, 'private_gpt.users.models.Role', 
                       os.path.join(fixtures_dir, 'roles.json'))
    
    user_roles = load_fixture(db_session, 'private_gpt.users.models.UserRole', 
                            os.path.join(fixtures_dir, 'user_roles.json'))
    
    print(f"Loaded {len(users)} users, {len(roles)} roles, {len(user_roles)} user_roles, {len(companies)} companies")
    return {
        'users': users,
        'departments': departments,
        'roles': roles,
        'user_roles': user_roles,
        'companies': companies
    }

# For pytest usage
@pytest.fixture
def pytest_load_all_fixtures(db_session):
    """Pytest fixture to load all fixtures for testing."""
    return load_all_fixtures(db_session)

if __name__ == "__main__":
    # When run directly as a script
    from private_gpt.users.db.session import SessionLocal
    
    # Create a new session
    session = SessionLocal()
    try:
        print("Loading fixtures...")
        load_all_fixtures(session)
        print("Fixtures loaded successfully")
    except Exception as e:
        print(f"Error loading fixtures: {e}")
        session.rollback()
        raise
    finally:
        session.close()