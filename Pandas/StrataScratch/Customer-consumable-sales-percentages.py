# Import your libraries
import pandas as pd

# Start writing code
# online_orders.head()
# online_products.product_class.unique()

df = online_orders.merge(online_products, how = 'left', on = 'product_id')

cond = (df['product_family'] == 'CONSUMABLE')
num_cus = df.loc[cond, 'customer_id'].value_counts()
print(num_cus)
