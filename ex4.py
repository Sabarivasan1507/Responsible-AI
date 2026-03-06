import json

database = {
    "user1": {"name": "Alice", "email": "alice@example.com", "data": "Activity log A"},
    "user2": {"name": "Bob", "email": "bob@example.com", "data": "Activity log B"}
}

def delete_user_data(user_id):
    """Handles the GDPR 'Right to Erasure'"""
    if user_id in database:
        del database[user_id]
        print(f"User {user_id} data deleted successfully.")
    else:
        print(f"Error: User {user_id} not found.")

def export_user_data(user_id):
    """Handles the GDPR 'Right to Access/Portability'"""
    if user_id in database:
        user_info = database[user_id]
       
        export_format = json.dumps(user_info, indent=4)
        print(f"Exporting data for {user_id}:")
        print(export_format)
        return export_format
    else:
        print(f"Error: User {user_id} not found.")
        return None


export_user_data("user2")

print("-" * 30)

delete_user_data("user1")

print(f"Current Database: {list(database.keys())}")