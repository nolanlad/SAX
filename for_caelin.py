from SAX import *
from total_sax_cluster import *
from sklearn.cluster import KMeans
import sklearn
import warnings
import gap
import glob
import pickle

NBINS = 5
space = EqualBinSpace(NBINS)
'''
batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Batch 1/*")
for f in batch1_fns:
    v1, v2 = read_ae_file2(f)
    X = get_heatmaps(v1,v2,space)
    scores = []
    silh = []
    for j in range(2,11):
        kmeans = KMeans(n_clusters=j, random_state=0).fit(X)
        sc = sklearn.metrics.davies_bouldin_score(X,kmeans.labels_)
        sc2 = sklearn.metrics.silhouette_score(X,kmeans.labels_)
        scores.append(sc)
        silh.append(sc2)
    plt.plot(range(2,11),scores)
    plt.plot(range(2,11),silh)
    plt.legend(["D-B","silh"])
    plt.savefig("scores_"+f[-5]+"_batch1.png")
    plt.show()
    
batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Batch 2/*")
for f in batch1_fns:
    v1, v2 = read_ae_file2(f)
    X = get_heatmaps(v1,v2,space)
    scores = []
    silh = []
    for j in range(2,11):
        kmeans = KMeans(n_clusters=j, random_state=0).fit(X)
        sc = sklearn.metrics.davies_bouldin_score(X,kmeans.labels_)
        sc2 = sklearn.metrics.silhouette_score(X,kmeans.labels_)
        scores.append(sc)
        silh.append(sc2)
    plt.plot(range(2,11),scores)
    plt.plot(range(2,11),silh)
    plt.legend(["D-B","silh"])
    plt.savefig("scores_"+f[-5]+"_batch2.png")
    plt.show()
'''

batch1_fns = glob.glob("./Raw waveforms/Raw waveforms/Data from Amjad/Fiber Tow/*")
for f in batch1_fns:
    v1, v2 = read_ae_file2(f)
    X = get_heatmaps(v1,v2,space)
    scores = []
    silh = []
    for j in range(2,11):
        kmeans = KMeans(n_clusters=j, random_state=0).fit(X)
        sc = sklearn.metrics.davies_bouldin_score(X,kmeans.labels_)
        sc2 = sklearn.metrics.silhouette_score(X,kmeans.labels_)
        scores.append(sc)
        silh.append(sc2)
    plt.plot(range(2,11),scores)
    plt.plot(range(2,11),silh)
    plt.legend(["D-B","silh"])
    plt.savefig("scores_"+f[-5]+"_batch2.png")
    plt.show()

leg = [s[-5] for s in batch1_fns]

'''
f = './Raw waveforms/Raw waveforms/Data from Amjad/Batch 1/HNSB1-1.txt'
v1, v2 = read_ae_file2(f)
X = get_heatmaps(v1,v2,space)

for i in range(200,600,100):
    scores = []
    for j in range(2,11):
        kmeans = KMeans(n_clusters=j, random_state=0).fit(X)
        sc = sklearn.metrics.davies_bouldin_score(X,kmeans.labels_)
        scores.append(sc)
    plt.plot(range(2,11),scores)
    plt.show()
'''
'''
gaps = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gaps = gap.gap(np.array(X),ks=range(2,11),nrefs=600)
plt.plot(range(2,11),gaps)
gaps = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gaps = gap.gap(np.array(X),ks=range(2,11),nrefs=800)
plt.plot(range(2,11),gaps)
'''
