import os
from openai import OpenAI
import pandas as pd
import pickle
from dotenv import load_dotenv
import numpy as np


def cosine_distance(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def main():

    # [ ] Store user searches in a small cache/db to optimize recommendations (basically recommend other products the user was previously interested in to provide better UX).
    # Implement an average of all the user history embeddings + current search for a query vector (implement a weighted average to prioritize current search compared to history (80-20?))
    # [ ] Create API that serves this recommendation engine alongside a basic frontend that makes queries in real time
    
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAPI_KEY'))

    fullProducts = pd.read_csv('newproducts.csv')
    fullProductsList = fullProducts.to_dict(orient='records')

    with open("newembeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    user_query = input('What product would you like to search for?\n')
    user_cap = int(input('How many similar results do you want to see?\n'))
    
    print('\n\nINITIALIZING\n\n')

    embedded_user_query = client.embeddings.create(
        input=user_query,
        model='text-embedding-3-small'
    ).model_dump()

    query_vector = embedded_user_query['data'][0]['embedding']

    cos_angles = []

    for index, embedding in enumerate(embeddings):
        cos_angles.append({'distance': cosine_distance(query_vector, embedding), 'index': index})

    
    sortedDistances = sorted(cos_angles, key=lambda x:x['distance'])[:user_cap]

    # print(sortedDistances)

    print('RESULTS FOUND'.center(30, '-'), '\n\n')

    for distance in sortedDistances:
        print('\n', 'PRODUCT DETAILS:'.center(30, '~'))
        print()

        print(f"""
Product Name: {fullProductsList[distance['index']]['Product Name']}

Category: {fullProductsList[distance['index']]['Category']}

Description: {fullProductsList[distance['index']]['description']}

Product Link: {fullProductsList[distance['index']]['Product Url']}

Price: {fullProductsList[distance['index']]['Selling Price']}

{'=' * 40}
""")



if __name__ == '__main__':
    main()
