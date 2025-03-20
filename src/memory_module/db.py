from pymongo import MongoClient
from copy import deepcopy

# Database setup
client = MongoClient("mongodb://localhost:27017/")
db = client["restaurant_db"]
customers = db["customers"]

def get_customer_profile(customer_id):
    """
    Retrieves customer profile if exists; otherwise, creates a new customer entry.

    Args:
        customer_id (str): Unique identifier of the customer.

    Returns:
        dict: Customer data.
    """
    customer = customers.find_one({"customerID": customer_id})

    if customer is None:
        # Customer doesn't exist, create a new entry
        new_customer = {
            "customerID": customer_id,
            "data": {}
        }
        customers.insert_one(new_customer)
        print(f"Created new customer '{customer_id}'.")
        return new_customer["data"]
    else:
        print(f"Retrieved existing customer '{customer_id}'.")
        return customer["data"]

# Main function
def update_customer_data(customer_id, new_data):
    customers.update_one(
        {"customerID": customer_id},
        {"$set": {"data": new_data}}
    )
    print(f"Updated existing entry for customer {customer_id}.")

# # Usage Example:
# customer_id = "alex"
# new_json = {
#     "data": {
        
#     }
# }

# update_customer_data(customer_id, new_json)