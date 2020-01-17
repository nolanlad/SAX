from SAX import *
from sklearn.cluster import KMeans
import sklearn

v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
#5 bins, anything > 7 led to overfitting of the data, bad test results
NBINS = 5
#space = EqualBinSpace(NBINS)
space = GaussBinSpace(NBINS)
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


'''
inert = []
silh = []
for i in range(2,10):
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    inert.append(kmeans.inertia_)
    silh.append(sklearn.metrics.silhouette_score(X,kmeans.labels_))
'''
def get_heatmaps(v1,v2,space):
    X = []
    #v1 = sax_normalize(v1)
    #v2 = sax_normalize(v2)
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


kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
lab = kmeans.labels_

v1, v2 = read_ae_file2("/home/nolan/Desktop/1002181-1.txt")
X2 = []
for i in range(0,len(v1)):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X2.append(space.to_vect(heatmap))

v1,v2 = read_ae_file2("./NASAFS/NHDT1.txt")

X3 = []
for i in range(0,len(v1)):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X3.append(space.to_vect(heatmap))


def pct_cluster(labels):
    return np.bincount(labels)/len(labels)


v1,v2 = read_ae_file2("./NASAFS/NHDT2.txt")

X4 = []
for i in range(0,len(v1)):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X4.append(space.to_vect(heatmap))
