import uuid
from sqlalchemy.orm import Session
from sqlalchemy import text
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.user import User
from private_gpt.users.models.department import Department
from private_gpt.users.models.category import Category
from private_gpt.users.models.company import Company
from private_gpt.users.models.role import Role as RoleModel
from private_gpt.users.models.user_role import UserRole
from private_gpt.users.core.security import get_password_hash
from private_gpt.users.constants.role import Role as RoleConst

def seed_data():
    db = SessionLocal()
    try:
        # 1. Clear existing data
        print("Clearing existing enterprise data...")
        tables = ["user_roles", "roles", "users", "departments", "categories", "companies"]
        for table in tables:
            db.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
        db.commit()

        # 2. Create Company
        print("Creating Company...")
        company = Company(name="NextGen Corp")
        db.add(company)
        db.flush()
        
        # 3. Create Roles
        print("Creating Roles...")
        roles_to_create = [RoleConst.GUEST, RoleConst.ADMIN, RoleConst.SUPER_ADMIN, RoleConst.OPERATOR]
        role_map = {}
        for r in roles_to_create:
            role = RoleModel(name=r["name"], description=r["description"])
            db.add(role)
            db.flush()
            role_map[r["name"]] = role

        # 4. Create Departments
        print("Creating Departments...")
        dept_names = [
            "Human Resources",
            "Legal & Compliance",
            "IT & Engineering",
            "Sales & Marketing",
            "Operations & Logistics"
        ]
        dept_map = {}
        for name in dept_names:
            dept = Department(name=name, company_id=company.id)
            db.add(dept)
            db.flush()
            dept_map[name] = dept

        # 5. Create Categories
        print("Creating Categories...")
        cat_names = [
            "Corporate Policies",
            "Standard Operating Procedures",
            "Technical Manuals",
            "Legal & Compliance",
            "Sales & Marketing Assets",
            "Confidential Project Docs"
        ]
        for name in cat_names:
            cat = Category(name=name)
            db.add(cat)
        
        # 6. Create Users (27 Actors + 1 Super Admin)
        print("Creating 27 Actors...")
        password = get_password_hash("GlobalPass2026!")
        
        actors = [
            # HR (5)
            ("anupama.subedi", "anupama.subedi@nextgen.com", "Human Resources"),
            ("priya.sharma", "priya.sharma@nextgen.com", "Human Resources"),
            ("jenny.anderson", "jenny.anderson@nextgen.com", "Human Resources"),
            ("ramesh.adhikari", "ramesh.adhikari@nextgen.com", "Human Resources"),
            ("deepak.adhikari", "deepak.adhikari@nextgen.com", "Human Resources"),
            # Legal (5)
            ("dikshya.thapa", "dikshya.thapa@nextgen.com", "Legal & Compliance"),
            ("amitabh.pandit", "amitabh.pandit@nextgen.com", "Legal & Compliance"),
            ("sarah.jones", "sarah.jones@nextgen.com", "Legal & Compliance"),
            ("suresh.gupta", "suresh.gupta@nextgen.com", "Legal & Compliance"),
            ("manisha.karki", "manisha.karki@nextgen.com", "Legal & Compliance"),
            # IT (6)
            ("bikash.poudel", "bikash.poudel@nextgen.com", "IT & Engineering"),
            ("rohit.singh", "rohit.singh@nextgen.com", "IT & Engineering"),
            ("lisa.chen", "lisa.chen@nextgen.com", "IT & Engineering"),
            ("prashat.acharya", "prashat.acharya@nextgen.com", "IT & Engineering"),
            ("rahul.verma", "rahul.verma@nextgen.com", "IT & Engineering"),
            ("bimal.thapa", "bimal.thapa@nextgen.com", "IT & Engineering"),
            # Sales (6)
            ("sneha.kapoor", "sneha.kapoor@nextgen.com", "Sales & Marketing"),
            ("biren.rana", "biren.rana@nextgen.com", "Sales & Marketing"),
            ("alex.perkins", "alex.perkins@nextgen.com", "Sales & Marketing"),
            ("nirupama.rai", "nirupama.rai@nextgen.com", "Sales & Marketing"),
            ("anjali.kumari", "anjali.kumari@nextgen.com", "Sales & Marketing"),
            ("pradeep.rai", "pradeep.rai@nextgen.com", "Sales & Marketing"),
            # Ops (5)
            ("sagar.magar", "sagar.magar@nextgen.com", "Operations & Logistics"),
            ("arjun.khadka", "arjun.khadka@nextgen.com", "Operations & Logistics"),
            ("tom.harris", "tom.harris@nextgen.com", "Operations & Logistics"),
            ("sunita.mahajan", "sunita.mahajan@nextgen.com", "Operations & Logistics"),
            ("kiran.tamang", "kiran.tamang@nextgen.com", "Operations & Logistics"),
        ]

        # Add a Super Admin back
        super_admin = User(
            username="super",
            email="superadmin@nextgen.com",
            hashed_password=get_password_hash("supersecretpassword"),
            is_active=True,
            company_id=company.id,
            department_id=dept_map["IT & Engineering"].id
        )
        db.add(super_admin)
        db.flush()
        db.add(UserRole(user_id=super_admin.id, role_id=role_map["SUPER_ADMIN"].id, company_id=company.id))

        # Track which departments already have an operator assigned
        operators_assigned = set()

        for username, email, dept_name in actors:
            user = User(
                username=username,
                email=email,
                hashed_password=password,
                is_active=True,
                company_id=company.id,
                department_id=dept_map[dept_name].id
            )
            db.add(user)
            db.flush()
            
            # Assign OPERATOR role to the first person in each department, GUEST to others
            role_name = "GUEST"
            if dept_name not in operators_assigned:
                role_name = "OPERATOR"
                operators_assigned.add(dept_name)
                print(f"Assigning OPERATOR role to {username} ({dept_name})")
            
            user_role = UserRole(
                user_id=user.id,
                role_id=role_map[role_name].id,
                company_id=company.id
            )
            db.add(user_role)
        
        db.commit()
        print(f"Successfully seeded simulation data: 1 Company, 5 Departments, 6 Categories, and 28 Users (including 5 Operators).")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
