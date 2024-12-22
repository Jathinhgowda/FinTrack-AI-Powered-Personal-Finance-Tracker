from app import db, User, Transaction

# Create database tables
db.create_all()

# Optional: Add a default user (for testing)
default_user = User(username='testuser', password='$2b$12$somethinghashedforpassword')  # Replace with hashed password
db.session.add(default_user)
db.session.commit()

print("Database initialized successfully!")
