from SAX import *
from total_sax_cluster import *
from sklearn.cluster import KMeans
import sklearn
import warnings
import gap
import os

spacef = EqualBinSpace

NBINS = 5
space = spacef(NBINS)
pca = sklearn.decomposition.KernelPCA(n_components=2)
pca.fit(X)

v11, v21 = read_ae_file2("./nasa-minicomp/1-2.txt")

def plot_foo(fname,col):
    v11, v21 = read_ae_file2(fname)
    X = get_heatmaps(v11,v21,space)
    X = np.array(X)
    Xto = pca.transform(X)
    #plt.plot(X[:,1],X[:,2],'o',color=col)
    plt.plot(Xto[:,0],Xto[:,1],'o',color=col)

def get_transform_data(fname):
    v11, v21 = read_ae_file2(fname)
    X = get_heatmaps(v11,v21,space)
    X = np.array(X)
    #Xto = pca.transform(X)
    return X

def plot_clusters(Xdata,labels,n_clusters):
    x = Xdata[:,0]
    y = Xdata[:,1]
    for n in range(n_clusters):
        plt.plot(x[labels==n],y[labels==n],'o')
        


plot_foo("./nasa-minicomp/1-1.txt",'red')
plot_foo("./nasa-minicomp/1-2.txt",'red')
#plt.show()
#plot_foo("./nasa-minicomp/2-1.txt",'red')
#plot_foo("./nasa-minicomp/2-2.txt",'red')
for f in os.listdir("./NASAFS"):
    fname = "./NASAFS/%s"%f
    plot_foo(fname,'blue')
plt.show()

'''
fname = "./nasa-minicomp/1-1.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)

pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')

############################################################
fname = "./nasa-minicomp/1-2.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')

############################################################
fname = "./nasa-minicomp/2-1.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')

############################################################
fname = "./nasa-minicomp/2-2.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')

fname = "./minicomp-dat/1-3.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)

#pca = sklearn.decomposition.PCA(n_components=2)
#pca.fit(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')


fname = "/home/nolan/Desktop/gg.txt"

v11, v21 = read_ae_file2(fname)
X = get_heatmaps(v11,v21,space)
X = np.array(X)
Xto = pca.transform(X)

plt.plot(Xto[:,0],Xto[:,1],'o')


plt.show()
'''
'''
for f in os.listdir("./NASAFS"):
    fname = "./NASAFS/%s"%f
    v12, v22 = read_ae_file2(fname)

    X = get_heatmaps(v12,v22,space)
    Xref = np.array(X)

    Xt = pca.transform(Xref)
    plt.plot(Xt[:,0],Xt[:,1],'o')

plt.show()
'''

