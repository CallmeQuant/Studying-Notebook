# Import your libraries
import pandas as pd

# Start writing code
# print(online_orders.head())
# print(online_sales_promotions.head())
# print(online_products.head())

# Merging three dataframes on common columns
total_df = online_orders.merge(online_sales_promotions, on = 'promotion_id').merge(online_products, on = 'product_id')

# Adding 'sales' column
total_df['sales'] = total_df['cost_in_dollars'].multiply(total_df['units_sold'])

# print(total_df.head())
# print(total_df['product_class'].nunique())
# # print(total_df['product_class'])
# print(total_df['media_type'].nunique())

# Create sales on category and media type dataframe
sales_per_cat_med = total_df.groupby(['media_type', 'product_class']).agg({'sales': 'sum'})

# print(sales_per_cat_med.head())

# First approach: 'groupby' on media type and compute sales per product class over total sales of media type
# Create sales on category within each media type dataframe
sales_per_cat = sales_per_cat_med.groupby(level = 1).apply(lambda x: 100 * x / float(x.sum())).rename(columns = {'sales': 'sales percent'})

# print(sales_per_cat)

# Second approach: Using 'pd.crosstab' function
sales_per_cat_2 = pd.crosstab(index=total_df['product_class'], columns=total_df['media_type'], values=total_df['sales'], aggfunc='sum', normalize='index').applymap('{:.2f}'.format)
 
# print(sales_per_cat_2)
