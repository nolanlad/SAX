from SAX import *
from total_sax_cluster import *
from sklearn.cluster import KMeans
import sklearn
import warnings
import gap
import pickle


v1,v2 = read_ae_file2("./NASAFS/NHDT1.txt")
#5 bins, anything > 7 led to overfitting of the data, bad test results
#NBINS = 5
NBINS = 5
space = EqualBinSpace(NBINS)
'''
X = []
for i in range(0,len(v1)):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X.append(space.to_vect(heatmap))

inert = []
silh = []
for i in range(2,11):
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    inert.append(kmeans.inertia_)
    silh.append(sklearn.metrics.silhouette_score(X,kmeans.labels_))
'''
'''
def get_heatmaps(v1,v2):
    X = []
    for i in range(0,len(v1)):
        sig1 = v1[i]
        sig2 = v2[i]
        sig = sig2
        if max(np.abs(sig1)) > max(np.abs(sig2)):
            sig = sig1
        word = to_word_bins(sig,space)
        heatmap = word_to_subword_space(word,space)
        X.append(space.to_vect(heatmap))
    return X
'''
def pct_cluster(labels):
    return np.bincount(labels)/len(labels)

v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
X = get_heatmaps(v1,v2,space)
Xref = X
inert = []
silh = []
'''
for i in range(2,11):
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    inert.append(kmeans.inertia_)
    silh.append(sklearn.metrics.silhouette_score(X,kmeans.labels_))

gaps = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gaps = gap.gap(np.array(Xref),ks=range(2,11),nrefs=200)
'''
n_cluster = 7

#kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
f = open('all_NASA_minicomp.pkl','rb')
Xmini = pickle.load(f)
f.close()
kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(Xmini)

tot = np.zeros(n_cluster)

v1,v2 = read_ae_file2("./NASAFS/NHDT1.txt")
X = get_heatmaps(v1,v2,space)
lab = kmeans.predict(X)
probs = pct_cluster(lab)
print(probs)
tot[:len(probs)]+=probs

v1,v2 = read_ae_file2("./NASAFS/NHDT2.txt")
X = get_heatmaps(v1,v2,space)
lab = kmeans.predict(X)
probs = pct_cluster(lab)
print(probs)
tot[:len(probs)]+=probs

v1,v2 = read_ae_file2("./NASAFS/NHDT3.txt")
X = get_heatmaps(v1,v2,space)
lab = kmeans.predict(X)
probs = pct_cluster(lab)
print(probs)
tot[:len(probs)]+=probs

for i in range(4,11):
    v1,v2 = read_ae_file2("./NASAFS/HNST%d.txt"%i)
    X = get_heatmaps(v1,v2,space)
    lab = kmeans.predict(X)
    probs = pct_cluster(lab)
    print(probs)
    tot[:len(probs)]+=probs

#print(tot/1)

labbs = kmeans.labels_
for i in range(n_cluster):
    x = Xmini[:,1][np.where(labbs == i)]
    y = Xmini[:,2][np.where(labbs == i)]
    plt.plot(x,y,'o',markersize=3)

plt.legend([str(i) for i in range(n_cluster)])
plt.show()
