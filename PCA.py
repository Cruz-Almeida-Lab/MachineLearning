import pandas as pd
from sklearn.decomposition import PCA

#designate input file
input_file = "pain_older.csv"

#pandas read input csv
dataset = pd.read_csv(input_file, header = 0,  sep=',')

#select data
X = dataset.iloc[:, 2:] #select columns 2 through end, predictors

#pca object
pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto')

pca.fit(X)

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)
