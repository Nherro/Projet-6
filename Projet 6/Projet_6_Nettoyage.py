##### Importation
from pandas_profiling import ProfileReport
import pandas as pd
import numpy as np
import seaborn as sns 
import tkinter
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

##### Fonction recheche outliers
#Utilisation des quartiles pour retrouver les valeurs aberrantes.
#percentile est un concept de statistique qui permet de déterminer la position d’une valeur par rapport à un groupe donné
#Q1 = 25 percentile des données
#Q3 = 75 percentile des données
def outliers(s):
    iqr = (np.quantile(s, 0.75))-(np.quantile(s, 0.25)) # Différence entre le Quartile supérieure et le Quartile inférieure
    upper_bound = np.quantile(s, 0.75)+(1.5*iqr) # Cherche les valeurs supérieur à Q3 + 1.5*IQR 
    lower_bound = np.quantile(s, 0.25)-(1.5*iqr) # Cherche les valeurs inférieur à Q1 - 1.5*IQR
    f = []
    for i in s:
        if i > upper_bound: #Si prix > Q3 valeurs abberantes
            f.append(i)
        elif i < lower_bound: #Si prix < Q2 valeurs abberantes
            f.append(i)
    sums = len(f) #Nombre de Outlier
    pros = len(f)/len(s)*100 # % de Outlier
    d = {'IQR':iqr,
         'Upper Bound':upper_bound,
        'Lower Bound':lower_bound,
        'Sum outliers': sums,'percentage outliers':pros}
    d = pd.DataFrame(d.items(),columns = ['sub','values'])
    return(d)

#### Donnée
##### Data customers
customers = pd.read_csv('customers.csv')
#print(customers.shape)
#customers.head()
#Profiling customers
#prof_customers = ProfileReport(customers)
#profile_customers  =  ProfileReport ( customers , title = "customers" )
#profile_customers

##### Data products
products = pd.read_csv('products.csv')
#print(products.shape)
#products.head()
#Profiling products
#prof_products = ProfileReport(products)
#profile_products  =  ProfileReport ( products , title = "products" )
#profile_products

#Visualisation valeur aberrante sur price
figure = plt.figure(figsize = (15, 7))
plt.boxplot(products['price'],notch = True,vert=0)
plt.show()
#Information sur les outliers
outliers(products.price)
#List des outliers
#valeurs_aberrantes = (products.loc[products['price'] > 46.99])
#valeurs_aberrantes.sort_values(by=['price'], ascending=False)

products.sort_values(by=['price']) #Verification prix inférieure à 0
products.drop(products.loc[products['id_prod']=='T_0'].index, inplace=True) #Supression ligne test

##### Data transactions
transactions = pd.read_csv('transactions.csv')
#print(transactions.shape)
#transactions.head()
#Profiling transactions
#prof_transactions = ProfileReport(transactions)
#profile_transactions  =  ProfileReport ( transactions , title = "transactions" )
#profile_transactions



#Supression des lignes de test
transactions.drop(transactions.loc[transactions['id_prod']=='T_0'].index, inplace=True)
#Visualisation des valeurs en doublon
df_double = transactions.groupby(['id_prod', 'client_id']).agg({'client_id':'count'})
print(df_double)
double = transactions[(transactions['id_prod']=='0_0') & (transactions['client_id']=='c_1052')]
print(double)
transactions['date'] = pd.to_datetime(transactions['date']).dt.date #Garde uniquement l'année le mois et le jour
transactions['date'] = pd.to_datetime(transactions.date, format='%Y-%m-%d') #Convertir la colonne date en datetime

#Separe le mois et l'année
transactions['Year'] = transactions['date'].dt.year 
transactions['Month'] = transactions['date'].dt.month 
transactions['Day'] = transactions['date'].dt.day

#### Création data chiffre d'affaires
chiffre_daffaires = pd.merge(transactions, products, on  = "id_prod" ,how = 'left')
print(chiffre_daffaires.head())
#Profiling chiffre_daffaires
#prof_chiffre_daffaires = ProfileReport(chiffre_daffaires)
#profile_chiffre_daffaires =  ProfileReport ( chiffre_daffaires , title = "chiffre daffaires" )
#profile_chiffre_daffaires

#Visualisation des lignes où categ et price est null
nan_rows = chiffre_daffaires[(chiffre_daffaires['categ'].isnull()) & (chiffre_daffaires['price'].isnull())]
print(nan_rows)
print(nan_rows.groupby('id_prod').size())
print(chiffre_daffaires[(chiffre_daffaires['id_prod']=='0_2245')]) #Fiche produit manquant dans la table des produits

#Imputation moyenne
median = products['price'].median() #Calcul du prix median
chiffre_daffaires['price'].fillna(median, inplace=True)#Remplace le prix par le prix median
chiffre_daffaires['categ'].fillna(0.0, inplace=True) #Ajout du produit dans sa catégorie

#### Création data profil clients
profil_client = pd.merge(chiffre_daffaires, customers, on  = "client_id" ,how = 'left')
#profil_client.head()
#Profiling profil_client
#prof_profil_client = ProfileReport(profil_client)
#profile_profil_client =  ProfileReport ( profil_client , title = "profil clients" )
#profile_profil_client

#du principe que nous sommes en 2023
profil_client['birth'] = 2023 - profil_client['birth'] #Calcul de l'age des clients
profil_client.rename(columns={'birth':'Age'}, inplace=True) #renomme birth en age
profil_client['tranche_dage'] = pd.cut(profil_client['Age'], bins=[17, 29, 39, 49, 59, 100])
print(profil_client)


#### Suite analyse sur chiffre d'affaires Octobre 2021
octobre_2021 = chiffre_daffaires[(chiffre_daffaires['Year']== 2021) & (chiffre_daffaires['Month']== 10 )]
print(octobre_2021.head(10))
octobre_2021 = octobre_2021[['categ', 'Month', 'Year', 'Day']] #Selection des colonnes necessaires
octobre_2021.groupby('categ')['Day'].nunique().reset_index()
octobre_2021_categ = chiffre_daffaires[(chiffre_daffaires['Year']== 2021) 
                                       & (chiffre_daffaires['Month']== 10 ) 
                                       & (chiffre_daffaires['categ']== 1.0 )]
print(octobre_2021_categ.groupby('Day').nunique()) #Catégorie 1 manquant en quasi totalité sur Octobre 2021



##### Supression octobre 2021
drop_octobre_21 = chiffre_daffaires[(chiffre_daffaires['Year']== 2021) 
                                                 & (chiffre_daffaires['Month']== 10 )].index
chiffre_daffaires.drop(drop_octobre_21 , inplace=True)
octobre_2021 = chiffre_daffaires[(chiffre_daffaires['Year']== 2021) & (chiffre_daffaires['Month']== 10 )]
print(octobre_2021)
#Sauvegarde data vers csv
#profil_client.to_csv('profil_client.csv')
#chiffre_daffaires.to_csv('chiffre_daffaires.csv')
