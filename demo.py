#%%
import pandas as pd
import os
from dotenv import load_dotenv
import time
import re


#%%
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

#%%
data_asset_name = 'hacker-news'
data_asset_version = '1'
data_asset = ml_client.data.get(name=data_asset_name, version=data_asset_version)
data_uri = data_asset.path

#%%
pandas_df = pd.read_parquet(data_uri)




#%%
def extract_top_domains_df(df:pd.DataFrame)->pd.DataFrame:
    return (
        df.loc[df["url"].notna()]
        .assign(
            domain=df["url"].apply(
                lambda x: re.findall(r"http[s]?://([^/]+)/", x)[0]
                if x and re.findall(r"http[s]?://([^/]+)/", x)
                else ""
            )
        )
        .query('domain != ""')
        .groupby("domain")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

#%%
