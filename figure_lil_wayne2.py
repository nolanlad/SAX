from SAX import *
from total_sax_cluster import *
import scipy.stats as st

NBINS = 5
space = EqualBinSpace(NBINS)
#space = EquiBinSpace(NBINS)

v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
X = get_heatmaps(v1,v2,space)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
lads = kmeans.labels_

v1, v2 = read_ae_file2("./minicomp-dat/1-3.txt")

X = get_heatmaps(v1,v2,space)
lads = kmeans.predict(X)

#NBINS = 5
#space = EqualBinSpace(NBINS)
#space = GaussBinSpace(NBINS)
#space = EquiBinSpace(NBINS)
avg1 = np.zeros((NBINS,NBINS))

fig,ax = plt.subplots(nrows=3,ncols=8)
cmap = 'inferno'

for j,i in enumerate(np.where(lads==4)[0][:4]):
    
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    im = ax[0][j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)


for j,i in enumerate(np.where(lads==1)[0][-4:]):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    ax[1][j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)

for j,i in enumerate(np.where(lads==1)[0][:4]):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    ax[2][j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)


v1, v2 = read_ae_file2("./minicomp-dat/2-4.txt")

X = get_heatmaps(v1,v2,space)
lads = kmeans.predict(X)

for j,i in enumerate(np.where(lads==4)[0][:4]):
    
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    im = ax[0][4+j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)


for j,i in enumerate(np.where(lads==1)[0][-4:]):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    ax[1][4+j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)

for j,i in enumerate(np.where(lads==1)[0][:4]):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    heatmap = -heatmap+1
    ax[2][4+j].imshow(heatmap,vmin=0, vmax=1,cmap=cmap)

#cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
#fig.colorbar(im,cax=cbar_ax)
plt.show()
