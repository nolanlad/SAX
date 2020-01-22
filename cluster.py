from SAX import *
from sklearn.cluster import KMeans

NBINS = 5
space = EqualBinSpace(NBINS)

v1,v2 = read_ae_file2("./minicomp-dat/2-4.txt")

X = get_heatmaps(v1,v2,space)

kmeans = KMeans(n_clusters=5,random_state=0).fit(X)
