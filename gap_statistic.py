#!/usr/bin/env python
"""
Author : Jaganadh Gopinadhan
Licence : Apahce 2
e-mail jaganadhg at gmail dot com 
"""
import scipy

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

import pandas as pd


class TWHGapStat(object):
    """
    Implementation of Gap Statistic from Tibshirani, Walther, Hastie to determine the 
    inherent number of clusters in a dataset with k-means clustering.
    Ref Paper : https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    
    def generate_random_data(self, X):
        """
        Populate reference data.
        
        Parameters
        ----------
        X : Numpy Array
            The base data from which random sample has to be generated
        
        Returns
        -------
        reference : Numpy Array
            Reference data generated using the Numpy/Scipy random utiity .
            NUmber of diamensions in the data will be as same as the base
            dataset. 
        """
        reference = scipy.random.random_sample(size=(X.shape[0], X.shape[1]))
        return reference
    
    def _fit_cluster(self,X, n_cluster, n_iter=5):
        """
        Fit cluster on reference data and return inertia mean.
        
        
        Parameters
        ----------
        X : numpy array
            The base data 
            
        n_cluster : int 
            The number of clusters to form 
            
        n_iter : int, default = 5
            number iterative lustering experiments has to be perfromed in the data.
            If the data is large keep it less than 5, so that the run time will be less.
        
        Returns
        -------
        mean_nertia : float 
            Returns the mean intertia value. 
        """
        iterations = range(1, n_iter + 1)
        
        ref_inertias = pd.Series(index=iterations)
        
        for iteration in iterations:
            clusterer = KMeans(n_clusters=n_cluster, n_init=3, n_jobs=-1)
            # If you are using Windows server n_jobs = -1 will be dangerous. So the 
            # value should be set to max cores - 3 . If we use all the cores available
            # in Windows server sklearn tends to throw memory error 
            clusterer.fit(X)
            ref_inertias[iteration] = clusterer.inertia_
        
        mean_nertia = ref_inertias.mean()
        
        return mean_nertia
    
    def fit(self,X,max_k):
        """
        Compute Gap Statistics
        Parameters
        ----------
        X : numpy array
            The base data 
        max_k :int 
            Maximum value to which we are going to test the 'k' in k-means algorithmn 
        Returns
        -------
        gap_stat : Pandas Series
            For eack k in max_k range gap stat value is returned as a Pandas Sereies.
            Index is K and valuess correspondes to gap stattistics for each K
        """
        
        k_range = range(1,max_k + 1)
        gap_stat = pd.Series(index=k_range)
        
        ref_data = self.generate_random_data(X)
        
        for k in k_range:
            base_clusterer = KMeans(n_clusters=k,n_init = 3, n_jobs = -1)
            base_clusterer.fit(X)
            
            ref_intertia = self._fit_cluster(ref_data,k)
            
            cur_gap = scipy.log(ref_intertia - base_clusterer.inertia_)
            
            gap_stat[k] = cur_gap
        
        return gap_stat

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data 
    
    gap_stat = TWHGapStat()
    gs = gap_stat.fit(X,5)
    print(gs)
