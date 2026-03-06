import datetime

user_consent = True
consent_purpose = "Marketing"
data_collection_purpose = "Marketing"

def process_user_data(user_id, data, purpose, consent):
    if not consent or purpose != data_collection_purpose:
        print("Error: Purpose mismatch or no consent.")
        return None

    print(f"Processing data for {purpose}...")
    return {"user_id": user_id, "data": data, "timestamp": datetime.datetime.now()}

user_data = {"name": "Alice", "email": "alice@example.com"}
processed = process_user_data("123", user_data, "Marketing", user_consent)