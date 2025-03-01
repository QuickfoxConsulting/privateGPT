import json
import os
import importlib
import pytest
from sqlalchemy.orm import Session

def load_fixture(session: Session, model_name: str, fixture_file: str):
    """
    Load fixtures from a JSON file into the database.
    
    Args:
        session: SQLAlchemy session
        model_name: Fully qualified model name (e.g., 'private_gpt.models.User')
        fixture_file: Path to the JSON file with fixture data
    """
    module_path, class_name = model_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Load data from file
    with open(fixture_file, 'r') as f:
        fixtures = json.load(f)
    
    # Create and add model instances
    instances = [model_class(**fixture) for fixture in fixtures]
    session.add_all(instances)
    session.commit()
    
    return instances

@pytest.fixture
def load_all_fixtures(db_session):
    """Load all fixtures for testing."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    users = load_fixture(db_session, 'private_gpt.models.User', 
                         os.path.join(fixtures_dir, 'users.json'))
    
    roles = load_fixture(db_session, 'private_gpt.models.Role', 
                         os.path.join(fixtures_dir, 'roles.json'))
    
    user_roles = load_fixture(db_session, 'private_gpt.models.UserRole', 
                              os.path.join(fixtures_dir, 'user_roles.json'))
    
    companies = load_fixture(db_session, 'private_gpt.models.Company', 
                             os.path.join(fixtures_dir, 'companies.json'))
    
    print(f"Loaded {len(users)} users, {len(roles)} roles, {len(user_roles)} user_roles, {len(companies)} companies")
    return {
        'users': users,
        'roles': roles,
        'user_roles': user_roles,
        'companies': companies
    }


if __name__ == "__main__":
    from private_gpt.users.db.session import SessionLocal    
    load_all_fixtures(SessionLocal())
