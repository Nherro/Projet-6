##### Importation
from pandas_profiling import ProfileReport
import pandas as pd
import numpy as np
import seaborn as sns 
import tkinter
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import scipy.stats as stats
matplotlib.use('TkAgg')
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None  # default='warn'

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
chiffre_daffaires = pd.read_csv('chiffre_daffaires.csv')
profil_client = pd.read_csv('profil_client.csv')
products = pd.read_csv('products.csv')
customers = pd.read_csv('customers.csv')
transactions = pd.read_csv('transactions.csv')
del chiffre_daffaires["Unnamed: 0"]
#chiffre_daffaires.head()
#octobre_2021 = chiffre_daffaires[(chiffre_daffaires['Year']== 2021) & (chiffre_daffaires['Month']== 10 )]
#octobre_2021.head(10)
del profil_client["Unnamed: 0"]
### Demande d'Antoine
#### Indicateurs et graphiques autour du chiffre d'affaires
##### Indicateur sur le prix
Tendance centrale
products.drop(products.loc[products['id_prod']=='T_0'].index, inplace=True) #Supression ligne test
print('Le prix moyen est de:          ', products.price.mean()) #Prix moyen
print('Le prix moyen median est de:   ', products.price.median()) #Prix median
print('Le prix le plus représenté est:', products.price.mode()) # Prix le plus représenté
print('Prix le plus élevé:            ',products.price.max()) #Prix max
print('Prix le plus bas:              ',products.price.min()) #Prix min
Tendance dispersion
st_dev_prix = np.std(products.price)
print('Le prix etendu est de:        ' ,products.price.max()-products.price.min()) #Etendu sur le prix
print('L\'ecart type du prix est de: ' ,str(st_dev_prix)) #Ecart-type
##### Indicateur sur l'age
#Tendance centrale
print('Age moyen est de:            ',(2023-customers.birth).mean()) #Age moyen
print('Age median est de:           ',(2023-customers.birth).median()) #Age median
print('Age le plus représenté est:  ',(2023-customers.birth).mode()) #Age le plus représenté
print('Age le plus élevé:           ',(2023-customers.birth).max()) #Age max
print('Age le plus bas:             ',(2023-customers.birth).min()) #Age min
#Tendance dispersion
st_dev_age = np.std(customers.birth)
print('L\'age etendu est de:           ', customers.birth.max()-customers.birth.min()) #Etendu sur l'age
print('L\'ecart type de l\'age est de: ', str(st_dev_age)) #Ecart-type
##### Chiffre d'affaires par année
ca_annee = chiffre_daffaires.groupby(['Year']
                                    ).agg(ca_annee =('price', 'sum')).reset_index()
ca_annee_categ = chiffre_daffaires.groupby(['Year', 'categ']
                                          ).agg(ca_annee_categ =('price', 'sum')).reset_index()
#ca_annee_categ.sort_values(by=['ca_annee_categ'], ascending=False)
#ca_annee_categ
#ca_annee.to_csv('ca_annee.csv')
axis = ca_annee.plot.bar(rot=0, figsize=(10,5),x="Year", y="ca_annee" )
plt.title('Ca par année en Millions')
plt.xlabel('Année')
plt.ylabel('CA')
plt.show()
##### Chiffre d'affaires par mois
ca_mois = chiffre_daffaires.groupby(['Year', 'Month']
                                   ).agg(ca_mois =('price', 'sum')).reset_index()
#ca_mois
#ca_mois.to_csv('ca_mois.csv')
df = sns.load_dataset('tips') 
df = chiffre_daffaires.groupby(['Year', 'Month']
                              ).agg(ca_mois =('price', 'sum')).reset_index()
sns.barplot(x="Month", 
           y="ca_mois", 
           hue="Year", 
           data=df) 
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ], 
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Chiffre d\'affaires par mois')
plt.show()
##### Chiffre d'affaires par mois et catégories
ca_mois_categ = chiffre_daffaires.groupby(['Year', 'Month', 'categ']
                                         ).agg(ca_mois_categ =('price', 'sum')).reset_index()
monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
ca_mois_categ['Month'] = ca_mois_categ['Month'].map(monthDict)
#ca_mois_categ.head()
#ca_mois_categ.sort_values(by=['ca_mois_categ'], ascending=False)
##### Chiffre d'affaires par jour
ca_jour = chiffre_daffaires.groupby(['Year', 'Month', 'Day']
                                   ).agg(ca_jour =('price', 'sum')).reset_index()
#ca_jour
ca_jour_categ = chiffre_daffaires.groupby(['Year', 'Month', 'Day', 'categ']
                                         ).agg(ca_jour_categ =('price', 'sum')).reset_index()
#ca_jour_categ.sort_values(by=['ca_jour_categ'], ascending=False)
#ca_jour_categ
st_dev_ca = np.std(ca_jour.ca_jour)
# Tendance centrale
print('Le CA median est de:  '         ,ca_jour.ca_jour.median())
print('Le CA le plus représenté est:' ,ca_jour.ca_jour.mode().mean())
#Tendance dispersion
print('Le CA etendu est de: ca      ' ,ca_jour.ca_jour.max()-ca_jour.ca_jour.min())
print('L\'ecart type du CA est de: '  ,str(st_dev_ca))

##### Moyenne mobile par année
moyenne_mobile = chiffre_daffaires.groupby(['Year','Month']
                                          ).agg(ca =('price', 'sum')).reset_index()
#moyenne_mobile
moyenne_mobile['ca'] = moyenne_mobile['ca'] / 30
moyenne_mobile['Month'] = moyenne_mobile['Month'].map(monthDict)
moyenne_mobile = moyenne_mobile.assign(Date = moyenne_mobile.Year.astype(str) +', ' + moyenne_mobile.Month.astype(str))
moyenne_mobile.set_index('Month', inplace=True)#Utilisation de la date comme index
sma2021 = moyenne_mobile[moyenne_mobile['Year']== 2021]
sma2022 = moyenne_mobile[moyenne_mobile['Year']== 2022]
sma2023 = moyenne_mobile[moyenne_mobile['Year']== 2023]
sma2021 = pd.DataFrame(sma2021, columns=['ca'])
sma2022 = pd.DataFrame(sma2022, columns=['ca'])
sma2023 = pd.DataFrame(sma2023, columns=['ca'])
#moyenne_mobile.head()
#moyenne_mobile.to_csv('moyenne_mobile.csv')
plt.figure(figsize = (15,5))
plt.title('Moyenne Mobile')
plt.plot(sma2022, label="2022")
plt.plot(sma2021, label="2021")
plt.plot(sma2023, label="2023")
matplotlib.pyplot.xticks(fontsize=10)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.xlabel("Month")
plt.ylabel("Moyenne mobile CA")
plt.legend()
plt.show()
##### Moyenne mobile total
moyenne_mobile.set_index('Date', inplace=True)#Utilisation de la date comme index
sma = moyenne_mobile['ca']#Moyenne mobile total
plt.figure(figsize = (15,5))
plt.title('Moyenne Mobile')
plt.plot(sma, label="sma Total")
matplotlib.pyplot.xticks(fontsize=10)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.xlabel("Date")
plt.ylabel("Moyenne mobile CA")
plt.legend()
plt.show()
##### Chiffre_d'affaire par catégorie
chiffre_daffaires_categ = chiffre_daffaires.groupby(['categ']
                                                   ).agg(ca_categ = ('price', 'sum')).reset_index()
#chiffre_daffaires_categ.to_csv('chiffre_daffaires_categ.csv')
ax = chiffre_daffaires_categ.plot(x="categ", y="ca_categ", kind="bar", rot=0) 
plt.title('Chiffre d\'affaires par catégorie')
plt.show()
##### Top et flop produits
#Top produits CA
top_ca = chiffre_daffaires.groupby(['id_prod']
                               ).agg(ca_prod = ('price', 'sum')).reset_index()
#top_ca.sort_values(by=['ca_prod'], ascending=False).head()
#Flop produits CA
flop_ca = chiffre_daffaires.groupby(['id_prod']
                                ).agg(ca_prod = ('price', 'sum')).reset_index()
#flop_ca.sort_values(by=['ca_prod']).head()
#Top produits vente
top_vente = chiffre_daffaires.groupby(['id_prod']
                               ).agg(vente = ('session_id', 'count')).reset_index()
#top_vente.sort_values(by=['vente'], ascending=False).head()
#Flop produits vente
flop_vente = chiffre_daffaires.groupby(['id_prod']
                                ).agg(vente = ('price', 'sum')).reset_index()
#flop_vente.sort_values(by=['vente']).head()
##### Chiffre d'affaires par tranche d'age
indicateur_vente_age = profil_client.groupby(['tranche_dage']
                                            ).agg(ca =('price', 'sum')).reset_index()
#indicateur_vente_age.sort_values(by=['ca'], ascending=False)
##### Chiffre d'affaires par genre
indicateur_vente_genre = profil_client.groupby(['sex']
                                              ).agg(ca =('price', 'sum')).reset_index()
#indicateur_vente_genre.head()
##### Total achat par client
indicateur_vente = profil_client.groupby(['client_id', 'sex', 'Age', 'tranche_dage']
                                        ).agg(total_achat =('price', 'sum')).reset_index()
#indicateur_vente.sort_values(by=['total_achat'], ascending=False)
c_1609 = profil_client[(profil_client['client_id']== 'c_1609') ]
#c_1609.groupby('id_prod').size()
##### Courbe de Lorenz
def split_list(a_list):
    half = len(a_list)//10
    return a_list[:half], a_list[half:]
 ##### Par tranche d'age
indicateur_vente_age = indicateur_vente_age.sort_values(by=['ca']) #Classement par ordre croissant
indicateur_vente_age['lorenz'] = indicateur_vente_age['ca'].cumsum() / indicateur_vente_age['ca'].sum() # Valeur maximal à 1
#indicateur_vente_age
courbe_lorenz = indicateur_vente_age[['tranche_dage', 'lorenz']] #Selection des colonnes necessaires
#courbe_lorenz.head()
courbe_lorenz.set_index('tranche_dage', inplace=True) #Utilisation de tranche_dage comme index
axis = courbe_lorenz.plot(rot=0, figsize=(10, 5))
plt.title('Courbe de Lorenez par tranche d\'age')
plt.xlabel('Tranche_dage')
plt.ylabel('CA')
plt.show()
 ##### Par produit
lorenz_prod = chiffre_daffaires.groupby(['id_prod']).agg(ca_prod =( 'price', 'sum'))
lorenz_prod = lorenz_prod.sort_values(by=['ca_prod']).reset_index()
lorenz_prod['tranche'] = pd.qcut(lorenz_prod.index, 10, labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
#lorenz_prod.head()
lorenz_prod = lorenz_prod.groupby(['tranche']).agg(ca_prod_perc =( 'ca_prod', 'sum'))
lorenz_prod['lorenz'] = (lorenz_prod['ca_prod_perc'].cumsum() / lorenz_prod['ca_prod_perc'].sum()) * 100
lorenz_prod = lorenz_prod[['lorenz']] #Selection des colonnes necessaires
#lorenz_prod.to_csv('lorenz_prod.csv')
axis = lorenz_prod.plot(rot=0, figsize=(10, 5))
plt.title('Courbe de Lorenez en fonction des produits')
plt.xlabel('% Produit')
plt.ylabel('% CA')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.show()
 ##### Par client
lorenz_client = chiffre_daffaires.groupby(['client_id']).agg(ca_client =( 'price', 'sum'))
lorenz_client = lorenz_client.sort_values(by=['ca_client']).reset_index()
lorenz_client['tranche'] = pd.qcut(lorenz_client.index, 10, labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
#lorenz_prod.head()
lorenz_client = lorenz_client.groupby(['tranche']).agg(ca_prod_perc =( 'ca_client', 'sum'))
lorenz_client['lorenz'] = (lorenz_client['ca_prod_perc'].cumsum() / lorenz_client['ca_prod_perc'].sum()) * 100
#lorenz_client.head()
lorenz_client = lorenz_client[['lorenz']] #Selection des colonnes necessaires
#lorenz_client.to_csv('lorenz_client.csv')
axis = lorenz_client.plot(rot=0, figsize=(10, 5))
plt.title('Courbe de Lorenez en fonction des clients')
plt.xlabel('% Client')
plt.ylabel('% CA')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.show()
### Demande de Julie
##### Corrélation entre genre et catégorie
corr_genre_categ = profil_client.groupby(['client_id', 'sex', 'categ']
                                        ).agg(total_achat =('price', 'count')).reset_index()
#corr_genre_categ.to_csv('corr_genre_categ.csv')
#Information sur les outliers
#outliers(corr_genre_categ.total_achat)
drop_corr_genre_categ = corr_genre_categ[(corr_genre_categ['total_achat']>83)].index #Selection des outliers
corr_genre_categ.drop(drop_corr_genre_categ , inplace=True) #Suppression des outlier pour le test
##### Tableau contingence 
X = "categ" 
Y = "sex" 
contigency = corr_genre_categ[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
#contigency
#Cartes thermiques (2 variables qualitatives)
plt.figure(figsize=(12,8)) 
plt.title('Corrélation catégorie et genre')
sns.heatmap(contigency, annot=True, cmap="YlGnBu")
plt.show()
##### Calcul khi2
contigency['f0_f'] = round((contigency['f'] / contigency['Total']) * 100, 2) # frequence reel hommes
contigency['f0_m'] = round((contigency['m'] / contigency['Total']) * 100, 2) # frequence reel femmes
contigency['eft_f'] = round((contigency['Total'] * contigency.at[contigency.index[-1], 'f0_f']) /100, 2) # Effectif theorique femmes
contigency['eft_m'] = round((contigency['Total'] * contigency.at[contigency.index[-1], 'f0_m']) /100, 2) # Effectif theorique hommes
#contigency
contigency['intersection_f'] = ((contigency['f'] - contigency['eft_f'])**2 /contigency['eft_f']) # Intersection femmes
contigency['intersection_m'] = ((contigency['m'] - contigency['eft_m'])**2 /contigency['eft_m']) # Intersection hommes
#contigency
khi2 = contigency['intersection_f'].sum() + contigency['intersection_m'].sum()
khi2
# D'après la table de probabilité khi2 pour alpha = 5%
# Pour un degrés de liberté de 2
khi2c = 5.99
if (khi2 < khi2c): 
    print('On accepte H0, il n\'y a pas de dépendance entre le genre et la catégorie de produit acheté')
else :
    print('On rejette H0 et on accepte H1, il existe une dépendance entre le genre et la catégorie de produit acheté')
##### Corrélation entre age, total achats ,fréquence d’achat et panier moyen
etude_selon_age = (profil_client.groupby(['Age']).agg(ca=('price', 'sum'), #Ca par tranche d'age
                                                nombre_client = ('client_id', 'nunique'), #Effectif par age
                                                nombre_de_commande = ('session_id', 'count')           
                                                      ).reset_index())
etude_selon_age['panier_moyen'] = round(etude_selon_age['ca']/ etude_selon_age['nombre_de_commande'],2) #Panier moyen
etude_selon_age['frequence_dachat'] = round(etude_selon_age['nombre_de_commande']/ etude_selon_age['nombre_client'],2) #Fréquence d'achat
etude_selon_age['% ca'] = round((etude_selon_age['ca'] / etude_selon_age['ca'].sum())*100, 2) # % Ca
#etude_selon_age.head()
#Information sur les outliers
#outliers(etude_selon_age['% ca'])
#etude_selon_age.sort_values(by=['% ca'], ascending=False)
drop_etude_selon_age = etude_selon_age[(etude_selon_age['% ca']>5)].index #Suppression des ligne où le CA est >5%
etude_selon_age.drop(drop_etude_selon_age , inplace=True) #Suppression des outlier pour les test
### Correlations entre l'age et le montant total achat
p_age_total_achat = etude_selon_age[['Age', 'ca']]
#p_age_total_achat.to_csv('p_age_total_achat.csv')
##### Calcul de Pearson
x = p_age_total_achat['Age'] 
y = p_age_total_achat['ca']  
#Calcul des moyenne
m_age = x.mean()
m_ca = y.mean()
#Calcul 1/nombre d'observation
n = 1/y.count() 
#Calcul de la covariance
cov = round(n*(((x - m_age) * (y - m_ca)).sum()), 2)
#Calcul de la variance Age
var_x = round(n*(((x - m_age)**2 ).sum()), 2) 
#Calcul de la variance CA
var_y = round(n*(((y - m_ca)**2 ).sum()), 2)
# Calcul corralations de Pearson
rp = round(cov / (np.sqrt(var_x * var_y)),2)
rp
print ('Forte relation négative entre les variables : r de Pearson =', rp)
#Graphique en nuage (Deux variables quantitatives)
model = np.poly1d(np.polyfit(x, y, 1))
plt.title('''Correlations entre l\'age et le montant total achat \n
             Coeffeficient de correlation de Pearson : %.3f''' % rp)
polyline = np.linspace(18, 100, 100)
plt.scatter(x, y)
plt.plot(polyline, model(polyline))
plt.xlabel('Age')
plt.ylabel('Chiffre d\'affaires') 
plt.show()
##### Test de signification du r de Pearson
n = p_age_total_achat['ca'].count() #nombre d'observation
tcalcule = rp* (np.sqrt(n-2/1-rp))
tcalcule
#D'après la table de student, pour un seuil significatif de 5%
#Un degré de liberté de 71
tcritique = 1.9944
if (tcalcule > tcritique): 
    print('On accepte H0, il n\'y a pas de dépendance entre l\'age et le monant total d\'achat')
else :
    print('On rejette H0 et on accepte H1, il existe une dépendance entre l\'age et le monant total d\'achat')
### Correlations entre l'age et la frequence d'achat
etude_spearman = etude_selon_age[['Age', 'frequence_dachat']]
#etude_spearman.to_csv('etude_spearman.csv')
##### Calcul de Spearman
x = etude_spearman['Age'] 
y = etude_spearman['frequence_dachat']  
#Ajout des rang
etude_spearman['rang_x'] = etude_selon_age['Age'].rank()
etude_spearman['rang_y'] = etude_selon_age['frequence_dachat'].rank()
#Calcul de la différence des rang elevé au carré
etude_spearman['d²'] = (etude_spearman['rang_x'] - etude_spearman['rang_y'])**2
#Somme de d²
s_d = etude_spearman['d²'].sum()
#Calcul nom d'observation
n = etude_selon_age['Age'].count()
#Vérification sur le nombre de rang lié si >10%, utiliser Kendall
#etude_spearman[(etude_spearman['d²']== 0)]
#Calcul corrélation de Spearman
rs = round(1 -((6*s_d)/(n**3-n)),2)
rs
print ('Faible relation négative entre les variables : rs de Spearman =', rs)
#Graphique en nuage (Deux variables quantitatives)
model = np.poly1d(np.polyfit(x, y, 2))
plt.title('''Correlations entre l\'age et la frequence d\'achat \n
             Le coffeficient de correlation de Spearman\'s est de:%.3f''' %rs)

polyline = np.linspace(18, 100, 100)
plt.scatter(x, y)
plt.plot(polyline, model(polyline))
plt.xlabel('Age')
plt.ylabel('Frequence d\'achat')
plt.show()
##### Test de signification du r de Spearman
tcalcule = rs* (np.sqrt(n-2/1-rs))
tcalcule
# D'après la table de student, pour un seuil significatif de 5%
#Un degré de liberté de 71
tcritique = 1.9944
if (tcalcule > tcritique): 
    print('On accepte H0, il n\'y a pas de dépendance entre l\'age et la frequence d\'achat')
else :
    print('On rejette H0 et on accepte H1, il existe une dépendance entre l\'age et la frequence d\'achat')
### Correlations entre l'age et le panier moyen
p_age_panier_moyen = etude_selon_age[['Age', 'panier_moyen']]
#p_age_panier_moyen.to_csv('p_age_panier_moyen.csv')
##### Calcul de Spearman
x = p_age_panier_moyen['Age'] 
y = p_age_panier_moyen['panier_moyen'] 
#Ajout des rang
p_age_panier_moyen['rang_x'] = p_age_panier_moyen['Age'].rank()
p_age_panier_moyen['rang_y'] = p_age_panier_moyen['panier_moyen'].rank()
#Calcul de la différence des rang elevé au carré
p_age_panier_moyen['d²'] = (p_age_panier_moyen['rang_x'] - p_age_panier_moyen['rang_y'])**2
#Somme de d²
s_d = p_age_panier_moyen['d²'].sum()
#Calcul nom d'observation
n = p_age_panier_moyen['Age'].count()
#Calcul corrélation de Spearman
rs = round(1 -((6*s_d)/(n**3-n)),2)
rs
print ('Faible relation négative entre les variables : r de Spearman =', rs)
#Graphique en nuage (Deux variables quantitatives)
model = np.poly1d(np.polyfit(x, y, 3))
plt.title('''Correlations entre l\'age et le panier moyen \n 
             Coeffeficient de correlation de Spearman :%.3f''' %rs)
#add fitted polynomial line to scatterplot
polyline = np.linspace(18, 100, 50)
plt.scatter(x, y)
plt.plot(polyline, model(polyline))
plt.xlabel('Age')
plt.ylabel('Panier moyen')
plt.show()
##### Test de signification du r de Spearman
n = p_age_panier_moyen['panier_moyen'].count() #nombre d'observation
tcalcule = rs* (np.sqrt(n-2/1-rs))
tcalcule
# D'après la table de student, pour un seuil significatif de 5%
#Un degré de liberté de 71
tcritique = 1.9944
if (tcalcule > tcritique): 
    print('On accepte H0, il n\'y a pas de dépendance entre l\'age et le panier moyen')
else :
    print('On rejette H0 et on accepte H1, il existe une dépendance entre entre l\'age et le panier moyen')
##### Corrélation entre categorie acheté et tranche d'age
corr_prod_age = profil_client.groupby(['client_id', 'tranche_dage', 'categ', 'Age']
                                        ).agg(total =('price', 'count')).reset_index()
#corr_prod_age.to_csv('corr_prod_age.csv')
#Information sur les outliers
#outliers(corr_prod_age.total)
drop_corr_prod_age = corr_prod_age[(corr_prod_age['total']>83)].index
corr_prod_age.drop(drop_corr_prod_age , inplace=True)
##### Tableau contingence 
#Utilisation de tranche d'age, car des fréqences sont <5% un utilisant l'age.
X = "tranche_dage" #Variable qualitative
Y = "categ" #Variable qualitative
contigency = corr_prod_age[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
#contigency
#Cartes thermiques
plt.figure(figsize=(12,8)) 
plt.title('Corrélation catégorie et tranche d\'age')
sns.heatmap(contigency, annot=True, cmap="YlGnBu")
plt.show()
contigency['f0_categ_0'] = round((contigency[0.0] / contigency['Total']) * 100, 2) # % frequence reel de vente categ 0
contigency['f0_categ_1'] = round((contigency[1.0] / contigency['Total']) * 100, 2) # % frequence reel de vente categ 1
contigency['f0_categ_2'] = round((contigency[2.0] / contigency['Total']) * 100, 2) # % frequence reel de vente categ 2
contigency['eft_categ_0'] = round((contigency['Total'] * contigency.at[contigency.index[-1], 'f0_categ_0']) /100, 2) # Effectif theorique de vente categ 0
contigency['eft_categ_1'] = round((contigency['Total'] * contigency.at[contigency.index[-1], 'f0_categ_1']) /100, 2) # Effectif theorique de vente categ 1
contigency['eft_categ_2'] = round((contigency['Total'] * contigency.at[contigency.index[-1], 'f0_categ_2']) /100, 2) # Effectif theorique de vente categ 2
#contigency
contigency['intersection_categ_0'] = ((contigency[0.0] - contigency['eft_categ_0'])**2 /contigency['eft_categ_0']) # Intersection femmes
contigency['intersection_categ_1'] = ((contigency[1.0] - contigency['eft_categ_1'])**2 /contigency['eft_categ_1']) # Intersection femmes
contigency['intersection_categ_2'] = ((contigency[2.0] - contigency['eft_categ_2'])**2 /contigency['eft_categ_2']) # Intersection femmes
#contigency
khi2 = contigency['intersection_categ_0'].sum() + contigency['intersection_categ_1'].sum() + contigency['intersection_categ_2'].sum()
khi2
# D'après la table de probabilité khi2 pour alpha = 5%
# Pour un degrés de liberté de 8
khi2c = 15.51
if (khi2 < khi2c): 
    print('On accepte H0, il n\'y a pas de dépendance entre la categorie acheté et la tranche d\'age')
else :
    print('On rejette H0 et on accepte H1, il existe une dépendance entre la categorie acheté et la tranche d\'age')
