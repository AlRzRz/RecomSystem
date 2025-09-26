import pandas as pd
from datasets import load_dataset
import tiktoken

# [x] Create function that generates a csv of the original dataset appended with the embeddings
# [ ] Create embeddings for all products in the dataset

# Amazon product descriptions dataset
# df = pd.read_parquet("hf://datasets/philschmid/amazon-product-descriptions-vlm/data/train-00000-of-00001.parquet")


def formatDictToString(productDict):
    return f"""
Product Name: {productDict['Product Name']},
Category: {productDict['Category']},
Description: {productDict['description']}
"""

def generateEmbeddings(productList, encoding='cl100k_base'):
    
    formattedProducts = [formatDictToString(product) for product in productList]

    total_tokens = 0

    for product_description in formattedProducts:
        pass

    return total_tokens



def generateCSVAttachEmbeddings():

    df = pd.read_csv('trimmed.csv', index_col='Index')
    
    cleaned = df.dropna()
    # print(cleaned.shape)

    list_of_dicts = cleaned.to_dict(orient='records')
    # print(list_of_dicts[0].keys())

    print(generateEmbeddings(list_of_dicts))




if __name__ == '__main__':
    generateCSVAttachEmbeddings()


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