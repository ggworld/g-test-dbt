
name = ['Amir','Guy','Mytal','Yotam','Geva','Idan','Gil','Eldad']
products = ['pvva','pvvb','dc-if','app','ai-platform']
import random
import pandas as pd
with open('data/fact_d.csv','w') as of:
    df = pd.read_csv('data/cities.csv')
    for n,i in df.iterrows():
        for prod in products:
            of.write(f"{random.choice(name)},{i['City']},{prod}\n")
            
                   
