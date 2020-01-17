from SAX import *
from total_sax_cluster import *
from sklearn.cluster import KMeans
import sklearn
import warnings
import gap
import os
import glob
from scipy.stats import kde

spacef = EqualBinSpace

NBINS = 5
space = spacef(NBINS)

def get_transform_data(fname):
    v11, v21 = read_ae_file2(fname)
    X = get_heatmaps(v11,v21,space)
    X = np.array(X)
    return X

# batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Batch 1/*")
batch1_fns = glob.glob("./minicomp-dat/*")
# batch2_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Batch 2/*")
# batch1_fns.extend(batch2_fns)

X = None

for f in batch1_fns:
    if type(X) == np.ndarray:
        Xp = get_transform_data(f)
        X = np.concatenate((X,Xp),axis=0)
    else:
        X = get_transform_data(f)
    #plt.plot(X[:,1],X[:,2],'bo',markersize=2)
    #plt.show()

Xmini = X

x = X[:,1]
y = X[:,2]
k = kde.gaussian_kde([x,y])
nbins = 300
xi, yi = np.mgrid[0:1:nbins*1j, 0:1:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi2 = zi.reshape(xi.shape)
plt.imshow((zi2))
#plt.plot(y*nbins,x*nbins,'ro',markersize=1)

plt.savefig("kde_w_points.png")
plt.show()



Xb1 = X

batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Fiber Tow/*")[3:]

X = None

for f in batch1_fns:
    if type(X) == np.ndarray:
        Xp = get_transform_data(f)
        X = np.concatenate((X,Xp),axis=0)
    else:
        X = get_transform_data(f)
    #plt.plot(X[:,1],X[:,2],'bo',markersize=2)
    #plt.show()

Xtow = X
'''
scores1 = [];silh1 = []
for j in range(2,11):
    kmeans = KMeans(n_clusters=j, random_state=0).fit(Xtow)
    sc = sklearn.metrics.davies_bouldin_score(Xtow,kmeans.labels_)
    sc2 = sklearn.metrics.silhouette_score(Xtow,kmeans.labels_)
    scores1.append(sc)
    silh1.append(sc2)
scores2 = [];silh2 = []
for j in range(2,11):
    kmeans = KMeans(n_clusters=j, random_state=0).fit(Xmini)
    sc = sklearn.metrics.davies_bouldin_score(Xmini,kmeans.labels_)
    sc2 = sklearn.metrics.silhouette_score(Xmini,kmeans.labels_)
    scores2.append(sc)
    silh2.append(sc2)
'''
x = X[:,1]
y = X[:,2]
k = kde.gaussian_kde([x,y])
nbins = 300
#xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
xi, yi = np.mgrid[0:1:nbins*1j, 0:1:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi2 = zi.reshape(xi.shape)
plt.imshow((zi2))
#plt.plot(y*nbins,x*nbins,'ro',markersize=1)

plt.savefig("kde_w_points_tow.png")
plt.show()

'''
batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Batch 2/*")

X = None

for f in batch1_fns:
    if type(X) == np.ndarray:
        Xp = get_transform_data(f)
        X = np.concatenate((X,Xp),axis=0)
    else:
        X = get_transform_data(f)
    #plt.plot(X[:,1],X[:,2],'bo',markersize=2)
    #plt.show()
jkl = []
for i in range(2,11):
    
    kmeans =  KMeans(n_clusters=i)
    kmeans.fit(Xp)
    v = sklearn.metrics.davies_bouldin_score(Xp,kmeans.labels_)
    jkl.append(v)
    

labs = kmeans.predict(Xp)

jkl = np.array(jkl)/np.max(jkl)

gaps = gap.gaps_safe(Xp,nrefs=200)

plt.plot(range(2,11),jkl)
plt.plot(range(2,11),gaps)

plt.legend(['Davies-Bouldin index','Gap statistic'])
plt.xlabel('Number of Clusters')
plt.ylabel('Normalized Score')
plt.show()
'''
