import os
from openai import OpenAI
import pandas as pd
import pickle

def main():

    # [ ] Import csv with pandas and turn into list
    # [ ] Ask for user input, embed, and then compare similarities with embeddings in products
    # [ ] Retrieve top 5 products and return to user
    # [ ] Store user searches in a small cache/db to optimize recommendations (basically recommend other products the user was previously interested in to provide better UX).
    # Implement an average of all the user history embeddings + current search for a query vector (implement a weighted average to prioritize current search compared to history (80-20?))
    # [ ] Create API that serves this recommendation engine alongside a basic frontend that makes queries in real time
    
    client = OpenAI()
    # [ ] Make new completedemo.csv file
    fullProducts = pd.read_csv('products.csv')

    fullProductsList = fullProducts.to_dict(orient='records')

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    user_query = input('What product would you like to search for?\n')
    
    embedded_user_query = client.embeddings.create(
        input=user_query,
        model='text-embedding-3-small'
    )

    cos_angles = []

    for product in fullProductsList:
        pass







if __name__ == '__main__':
    main()
