###### import pandas as pd 
import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
 
# Step 1 Dataset 
data = { 
  'UserID': (1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5), 
  'ProductID': ('vivo', 'oppo', 'motorola', 'samsung', 'realme', 'Xiaomi', 
                'oneplus', 'Google Pixel', 'iphone', 'nokia', 'infix', 'lava', 'itel'), 
  'Standing': (5, 3, 2, 4, 5, 3, 4, 3, 5, 4, 2, 5, 4) 
} 
 
df = pd.DataFrame(data) 
print(", Sample Dataset,") 
print(df) 
 
user_item_matrix = df.pivot(index='UserID', columns='ProductID', values='Standing').fillna(0)
 
# Step 3 point- point Similarity 
point_pointSimilarity = cosine_similarity(user_item_matrix.T)
point_pointSimilarity = pd.DataFrame(point_pointSimilarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
 
# Step 4 Recommendation Functions 
def recommend_products(user_id, user_item_matrix, item_similarity_df, top_n=3):
    user_ratings = user_item_matrix.loc[user_id]
    scores = item_similarity_df.dot(user_ratings) / item_similarity_df.sum(axis=1)
    scores = scores.drop(user_ratings[user_ratings > 0].index)
    return scores.sort_values(ascending=False).head(top_n)
 
def recommend_for_new_user(rated_products, item_similarity_df, top_n=3):
    user_ratings = pd.Series(rated_products)
    scores = item_similarity_df.dot(user_ratings) / item_similarity_df.sum(axis=1)
    scores = scores.drop(user_ratings.index, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n)

# Step 5 stoner Input 
print(" === E-commerce Product Recommendation === ") 
choice = input(" Are you an being  stoner?( yes no)").strip().lower() 

if choice == 'yes': 
    user_id = int(input(" Enter your UserID( 1- 5): ")) 
    if user_id in user_item_matrix.index:
        recommendations = recommend_products(user_id, user_item_matrix, point_pointSimilarity, top_n=3)
        print(f" Top- 3 Recommended Products for stoner {user_id}: {list(recommendations.index)}") 
    else:
        print(" UserID not  set up. Please enter a new  stoner  rather.") 

elif choice == 'no':
    print(" Enter your conditions for products you have tried( 1- 5).") 
    print(f" Available products {list(user_item_matrix.columns)}") 
    rated_products = {} 
    while True:
        product = input(" ProductID( or 'done' to finish): ").strip() 
        if product.lower() == 'done':
            break 
        if product not in user_item_matrix.columns: 
            print(" Invalid ProductID. Try again.") 
            continue 
        standing = int(input(f" Standing for {product}( 1- 5): ")) 
        rated_products[product] = standing
    if rated_products:
        recommendations = recommend_for_new_user(rated_products, point_pointSimilarity, top_n=3)
        print(f" Top- 3 Recommended Products for You {list(recommendations.index)}") 
    else:
        print(" No conditions  handed. Can not  induce recommendations.") 
else:
    print(" Invalid choice. Please enter 'yes' or 'no'.")
