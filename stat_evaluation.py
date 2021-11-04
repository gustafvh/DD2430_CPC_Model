from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import stats
#
# TODO: Insert our dataset in this shape
#
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

'''
Example silhouette score
'''
# km = KMeans(n_clusters=3, random_state=42)
# km.fit_predict(X)
# score = silhouette_score(X, km.labels_, metric='euclidean')
# print('Silhouetter Score: %.3f' % score)


'''
Kolmogorov-Smirnov test 
'''
# add acc_list which should contain accuracy from each test 
acc_list = np.linspace(-15, 15, 9)
print("Computing Kolmogorov-Smirnov test for accuracy: ")
ks_score = stats.kstest(acc_list, 'norm')
print(ks_score)