from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#
# TODO: Insert our dataset in this shape
#
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

km = KMeans(n_clusters=3, random_state=42)
km.fit_predict(X)
score = silhouette_score(X, km.labels_, metric='euclidean')
print('Silhouetter Score: %.3f' % score)
