from SAX import *


v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
X = get_heatmaps(v1,v2)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
lads = kmeans.labels_

v1, v2 = read_ae_file2("./minicomp-dat/1-3.txt")

X = get_heatmaps(v1,v2)
lads = kmeans.predict(X)

NBINS = 5
space = GaussBinSpace(NBINS)
avg1 = np.zeros((NBINS,NBINS))
for i in np.where(lads==4)[0][:20]:

    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg1+= heatmap

avg1 = avg1/20

avg2 = np.zeros((NBINS,NBINS))
for i in np.where(lads==1)[0][:20]:
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg2+= heatmap

avg2 = avg2/20

fig,ax = plt.subplots(nrows=2,ncols=2)

ax[0][0].imshow(avg1)
ax[0][1].imshow(avg2)


v1, v2 = read_ae_file2("./minicomp-dat/2-4.txt")

X = get_heatmaps(v1,v2)
lads = kmeans.predict(X)

NBINS = 5
space = GaussBinSpace(NBINS)
avg1 = np.zeros((NBINS,NBINS))
for i in np.where(lads==4)[0][:20]:

    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg1+= heatmap

avg1 = avg1/20

avg2 = np.zeros((NBINS,NBINS))
for i in np.where(lads==1)[0][:20]:
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg2+= heatmap

avg2 = avg2/20

ax[1][0].imshow(avg1)
ax[1][1].imshow(avg2)
