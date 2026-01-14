
import pandas as pd

try:
    df = pd.read_csv('d:/Coding Central/Business_Optima_Assignment/all_reviews_merged.csv')
    print(f'Total Reviews: {len(df)}')
    print(f'Unique Users: {df["user_name"].nunique()}')
    print(f'Unique Courses: {df["course_id"].nunique()}')
    
    user_counts = df['user_name'].value_counts()
    multi_review_users = user_counts[user_counts > 1]
    print(f'Users with > 1 review: {len(multi_review_users)}')
    print(f'Percentage of multi-review users: {len(multi_review_users)/len(df["user_name"].unique())*100:.2f}%')
    
    print("\nTop 5 Courses by Review Count:")
    print(df['course_id'].value_counts().head())

except Exception as e:
    print(f"Error: {e}")
