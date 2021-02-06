
import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
from helpers.dataset_read_func import read_dataset
df_=pd.DataFrame()
df_=read_dataset(df_)
df=df_.copy()

# Verinin db'den alınması.

# credentials.
creds = {'user': 'synan',
         'passwd': 'haydegidelum',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'group5'}
# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
query = "select * from online_retail_2010_2011"
df_mysql = pd.read_sql_query(query, conn)

query = "show databases"
pd.read_sql_query(query, conn)

query = "select * from online_retail_2010_2011 limit 5"
pd.read_sql_query(query, conn)

query = "select * from online_retail_2010_2011"
df_mysql = pd.read_sql_query(query, conn)

df.head()
df_mysql.head()

df.info()
df_mysql.info()

df_mysql["InvoiceDate"] = pd.to_datetime(df_mysql["InvoiceDate"])
df_mysql.rename(columns={"CustomerID": "Customer ID"}, inplace=True)

######################################
# Görev 2: crm_data_prep Fonksiyonu ile Veri Ön İşleme Yapınız
######################################

from helpers.helpers import crm_data_prep
df_prep = crm_data_prep(df)

df_prep.head()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_prep)

######################################
# Görev 3: create_cltv_p Fonksiyonu ile Predictive CLTV Segmentlerini Oluşturunuz
######################################

from helpers.helpers import create_cltv_p
cltv_p = create_cltv_p(df_prep)

check_df(cltv_p)
cltv_p.head()

cltv_p.groupby("cltv_p_segment").agg({"count", "sum"})

######################################
# Görev 4: İstenilen segmentlere ait kullanıcı id'lerine göre veri setini indirgeyiniz.
######################################

# id'lerin alınması
a_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "A"].index
b_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "B"].index
c_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "C"].index

# bu id'lere göre df'lerin indirgenmesi
a_segment_df = df_prep[df_prep["Customer ID"].isin(a_segment_ids)]
b_segment_df = df_prep[df_prep["Customer ID"].isin(b_segment_ids)]
c_segment_df = df_prep[df_prep["Customer ID"].isin(c_segment_ids)]
a_segment_df.head()

######################################
# Görev 5: Her bir segment için birliktelik kurallarının üretilmesi
######################################

from helpers.helpers import create_invoice_product_df

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules

from helpers.helpers import create_rules
rules_a = create_rules(a_segment_df)
# frozen type veriyi düzenlemeye yarayan kod
product_a = int(rules_a["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

rules_b = create_rules(b_segment_df)
product_b = int(rules_b["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

rules_c = create_rules(c_segment_df)
product_c = int(rules_c["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

def check_id (stock_code):
    product_name = df_prep[df_prep["StockCode"] == stock_code][["Description"]].values(0).tolist()
    return print(product_name)

check_id(22916)

######################################
# Görev 6: Alman Müşterilere Segmentlerine Göre Öneriler
######################################

# cltv_p'nin çıktısı olan dataframe'e recommended_product adında bir değişken ekleyiniz.
# her bir segment için 1 tane ürün ekleyiniz.
# Yani müşteri hangi segmentte ise onun için yukarıdan gelen kurallardan birisini ekleyiniz.

cltv_p.head()

germany_ids = df_prep[df_prep["Country"]=="Germany"]["Customer ID"].drop_duplicates()

cltv_p["recommended_product"] = ""

cltv_p.loc[cltv_p.index.isin(germany_ids)]
#A segmentindeki alman müşteriler
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A")]
#Asegmentindeki alman müşterilere product a yı önerdik
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A"), "recommended_product"] = product_a
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A")]
#B segmentindeki alman müşterilere product b yi önerdik
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B"), "recommended_product"] = product_b
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B")]
#C segmentindeki alman müşterilere product c yi önerdik
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C"), "recommended_product"] = product_c
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C")]

cltv_p.loc[cltv_p.index.isin(germany_ids)]

cltv_p[cltv_p.index == 12471].head()

cltv_p.index.name = 'CustomerID'
#sql çıktısını alalım
cltv_p.to_sql(name='recommended_df',
              con=conn,
              if_exists='replace',
              index=True,
              index_label="CustomerID")