import pandas as pd
import numpy as np

data = {
    'UserID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 32, 18, 47, 30],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
              'david@example.com', 'eve@example.com'],
    'PurchaseHistory': [150.00, 45.50, 0.00, 1200.00, 210.25],
    'Location': ['NY', 'CA', 'TX', 'NY', 'CA'],
    'IPAddress': ['192.168.1.1', '10.0.0.2', '172.16.0.3', '192.168.1.4', '10.0.0.5']
}

df_raw = pd.DataFrame(data)

print("Original DataFrame:")
print(df_raw)