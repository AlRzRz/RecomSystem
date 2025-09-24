import os
from openai import OpenAI
import pandas as pd

def main():
    # Amazon product descriptions dataset
    df = pd.read_parquet("hf://datasets/philschmid/amazon-product-descriptions-vlm/data/train-00000-of-00001.parquet")
    

if __name__ == '__main__':
    main()
