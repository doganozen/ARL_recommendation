import pandas as pd
def read_dataset(dataframe):
    dataframe = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
    return dataframe