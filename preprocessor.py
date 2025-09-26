import pandas as pd
from datasets import load_dataset
import tiktoken
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import pickle
import time


# Amazon product descriptions dataset
# df = pd.read_parquet("hf://datasets/philschmid/amazon-product-descriptions-vlm/data/train-00000-of-00001.parquet")
# [ ] Currently only using 70 products. Actuall dataset was 120522 tokens. Subset into smaller lists and send the embedding requests (if it was necessary to use the whole dataset)


def formatDictToString(productDict):
    return f"""
Product Name: {productDict['Product Name']},
Category: {productDict['Category']},
Description: {productDict['description']}
"""


def split_into_chunks(lst, chunk_size=60):

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]




def generateEmbeddings(productList, encoding='cl100k_base'):

    formattedProducts = [formatDictToString(product) for product in productList]

    total_tokens = 0
    encoding = tiktoken.get_encoding(encoding)

    for product_description in formattedProducts:
        tokens = encoding.encode(product_description)
        total_tokens += len(tokens)
    print(f'Total tokens in current productList: {total_tokens}')

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAPI_KEY'))
    
    splits = split_into_chunks(formattedProducts)

    all_embeddings = []

    for split in splits:
        time.sleep(0.4)
        embeddingModelResp = client.embeddings.create(
            input=split,
            model='text-embedding-3-small'
        ).model_dump()

        print(embeddingModelResp)
        final_embeddings = [item['embedding'] for item in embeddingModelResp['data']]
        all_embeddings.extend(final_embeddings)
    
    return all_embeddings



def generateCSVAttachEmbeddings():

    df = pd.read_csv('trimmed.csv', index_col='Index')
    
    # Using a subset of 70 products to stay below embedding token limit (testing purposes)
    # cleaned = df.dropna().iloc[:70].copy()
    cleaned = df.dropna().copy()


    records = cleaned.to_dict(orient='records')

    product_embeddings = generateEmbeddings(records)

    with open("newembeddings.pkl", "wb") as f:
        pickle.dump(product_embeddings, f)

    cleaned.to_csv('newproducts.csv')


def testing():
    df = pd.read_csv('trimmed.csv', index_col='Index')

    print(df.shape)




if __name__ == '__main__':
    generateCSVAttachEmbeddings()
    # testing()












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