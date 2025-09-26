import pandas as pd
from datasets import load_dataset
import tiktoken
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import pickle


# Amazon product descriptions dataset
# df = pd.read_parquet("hf://datasets/philschmid/amazon-product-descriptions-vlm/data/train-00000-of-00001.parquet")


def formatDictToString(productDict):
    return f"""
Product Name: {productDict['Product Name']},
Category: {productDict['Category']},
Description: {productDict['description']}
"""


def generateEmbeddings(productList, encoding='cl100k_base'):

    # [ ] Currently only using 70 products. Actuall dataset was 120522 tokens. Subset into smaller lists and send the embedding requests (if it was necessary to use the whole dataset)
    formattedProducts = [formatDictToString(product) for product in productList]

    total_tokens = 0
    encoding = tiktoken.get_encoding(encoding)

    for product_description in formattedProducts:
        tokens = encoding.encode(product_description)
        total_tokens += len(tokens)

    print(f'Total tokens in current productList: {total_tokens}')

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAPI_KEY'))
    
    embeddingModelResp = client.embeddings.create(
        input=formattedProducts,
        model='text-embedding-3-small'
    ).model_dump()

    print(embeddingModelResp)
    final_embeddings = [item['embedding'] for item in embeddingModelResp['data']]
    
    return final_embeddings



def generateCSVAttachEmbeddings():

    df = pd.read_csv('trimmed.csv', index_col='Index')
    
    # Using a subset of 70 products to stay below embedding token limit (testing purposes)
    cleaned = df.dropna().iloc[:70].copy()

    records = cleaned.to_dict(orient='records')

    product_embeddings = generateEmbeddings(records)

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(product_embeddings, f)

    cleaned.to_csv('products.csv')




if __name__ == '__main__':
    generateCSVAttachEmbeddings()








    # Early commented set-up
    
    # ds = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
    # df = ds.to_pandas()
    # df.to_csv('out.csv')
    # df = pd.read_csv('out.csv')
    # print(df.shape)
    # df_trimmed = df[['Product Name', 'Category', 'Selling Price', 'Product Url', 'description']]
    # df_trimmed.to_csv('trimmed.csv')
    # print(df.columns)
    # print(df.iloc[:, 0].name)
    # new = df.drop(df.iloc[:, 0].name, axis=1)
    # print(new.columns)
    # rows_with_any_na = df[df.isnull().any(axis=1)]
    # print(rows_with_any_na)