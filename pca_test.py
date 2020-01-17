from SAX import *
from total_sax_cluster import *
from sklearn.cluster import KMeans
import sklearn
import warnings
import gap

spacef = EqualBinSpace
'''
for i in range(3,10):
    NBINS = i
    print(i)
    #space = EqualBinSpace(NBINS)
    space = spacef(NBINS)

    v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
    X = get_heatmaps(v1,v2,space)
    Xref = np.array(X)

    pca = sklearn.decomposition.PCA(n_components=2)

    pca.fit(Xref)

    c = pca.get_covariance()

    eigvals = np.linalg.eigvals(c)
    print(np.linalg.eigvals(c))

    #Xt = pca.transform(Xref)

    #kmeans1 = KMeans(n_clusters=2, random_state=0).fit(Xref)
    #kmeans2 = KMeans(n_clusters=2, random_state=0).fit(Xt)

    #plt.plot(kmeans1.labels_,'bo')
    #plt.plot(kmeans2.labels_,'ro')
    #plt.show()
'''
NBINS = 5
#print(i)
#space = EqualBinSpace(NBINS)
space = spacef(NBINS)

v10, v20 = read_ae_file2("/home/nolan/Desktop/gg.txt")
v12, v22 = read_ae_file2("./minicomp-dat/1-3.txt")
v11, v21 = read_ae_file2("./minicomp-dat/2-4.txt")
v1 = v11 + v12 
v2 = v21 + v22 
X = get_heatmaps(v1,v2,space)
Xref = np.array(X)

pca = sklearn.decomposition.PCA(n_components=2)

pca.fit(Xref)

c = pca.get_covariance()

eigvals = np.linalg.eigvals(c)
print(np.linalg.eigvals(c))

Xt = pca.transform(Xref)

nclust = 2

kmeans1 = KMeans(n_clusters=nclust, random_state=0).fit(Xref)
kmeans2 = KMeans(n_clusters=nclust, random_state=0).fit(Xt)

print('score = ',np.sum(kmeans1.labels_==kmeans2.labels_)/len(Xt))

Xt1 = Xt[:,0]; Xt2 = Xt[:,1]
#Xt1 = Xref[:,1]; Xt2 = Xref[:,2]
xlab = kmeans2.labels_

for i in range(nclust):
    plt.plot(Xt1[xlab == i],Xt2[xlab == i],'o')

# plt.plot(Xt1[xlab == 1],Xt2[xlab == 1],'o')
# plt.plot(Xt1[xlab == 2],Xt2[xlab == 2],'o')
# plt.plot(Xt1[xlab == 3],Xt2[xlab == 3],'o')
# plt.plot(Xt1[xlab == 4],Xt2[xlab == 4],'o')
#plt.plot(kmeans2.labels_,'ro')
plt.show()