from SAX import *
from sklearn.cluster import KMeans

v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
#5 bins, anything > 7 led to overfitting of the data, bad test results
NBINS = 5
space = EquiBinSpace(NBINS)

X = []
for i in range(0,25):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X.append(space.to_vect(heatmap))


for i in range(-2,-27,-1):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    X.append(space.to_vect(heatmap))


#devide into two clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

mcracks = kmeans.labels_[:25]

fbreaks = kmeans.labels_[25:]

v1, v2 = read_ae_file2("/home/nolan/Desktop/1002181-1.txt")

print("Fraction correctness, test data")

Xtest = []

for i in range(0,20):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    Xtest.append(space.to_vect(heatmap))

test = kmeans.predict(Xtest)

print(sum(test)/len(test))
Xtest2 = []
v1, v2 = read_ae_file2("/home/nolan/Desktop/2-4.txt")
for i in range(0,20):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    Xtest2.append(space.to_vect(heatmap))

test2 = kmeans.predict(Xtest2)

print(sum(test2)/len(test2))
