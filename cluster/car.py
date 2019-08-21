# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:49:24 2019

@author: hujia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import kurtosis
from itertools import product
from collections import defaultdict
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as SRP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import FactorAnalysis as FA

#################################################

data = pd.read_csv("./car.csv")
y=data.iloc[:,[6]]
x0=data.iloc[:,[0]]
x1=data.iloc[:,[1]]
x2=data.iloc[:,[2]]
x3=data.iloc[:,[3]]
x4=data.iloc[:,[4]]
x5=data.iloc[:,[5]]

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
x0=le.fit_transform(x0)
x1=le.fit_transform(x1)
x2=le.fit_transform(x2)
x3=le.fit_transform(x3)
x4=le.fit_transform(x4)
x5=le.fit_transform(x5)
X=np.vstack((x0, x1, x2, x3, x4, x5)).transpose()
print(X)
scaler = MinMaxScaler(feature_range=[0,100])
scaler.fit(X)
X_norm = pd.DataFrame(scaler.transform(X))

#################################################
#K means clustering
def run_Kmeans(X_norm,y,title):
    range_n_clusters = [2,3,4,5,6]
    loss1 = []
    silhouette_avg1 = []
    homo1 = []
    comp1 = []
    NMI1 = []
    start = time.perf_counter() 
    for index, n_clusters in enumerate(range_n_clusters):
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer1 = KMeans(n_clusters=n_clusters, random_state=10).fit(X_norm)
        loss1.append(clusterer1.inertia_)    
        cluster1_labels = clusterer1.labels_
        
        clusterer2 = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_norm) 
        cluster2_labels = clusterer2.predict(X_norm)
        
        print(cluster1_labels)
        print(cluster2_labels)
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg1.append(silhouette_score(X_norm, cluster1_labels))
        homo1.append(metrics.homogeneity_score(y, cluster1_labels))
        comp1.append(metrics.completeness_score(y, cluster1_labels))
        NMI1.append(normalized_mutual_info_score(y, cluster1_labels))
    end = time.perf_counter() 
    print("Kmeans run time: %.1f [s]" % (start-end))
    #Kmeans
    plt.plot(range_n_clusters, loss1)
    plt.ylabel('SSE')
    plt.xlabel('number of cluster')
    plt.title(title)
    plt.show()
    
    plt.plot(range_n_clusters, silhouette_avg1, label="silhouette")
    plt.plot(range_n_clusters, homo1, label="homogeneity")
    plt.plot(range_n_clusters, comp1, label="completeness")
    plt.plot(range_n_clusters, NMI1, label="NMI")
    plt.ylabel('value')
    plt.xlabel('number of cluster')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    
    #visulization of clusters
    k1 = 4
    plt.figure()
    plt.hist(clusterer1.labels_, bins=np.arange(0, k1 + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k1))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.show()

#EM
def run_EM(X_norm,y,title):
    range_n_clusters = [2,3,4,5,6]
    silhouette_avg2 = []
    homo2 = []
    comp2 = []
    NMI2 = []
    AIC = []
    BIC = []
    start = time.perf_counter() 
    for index, n_clusters in enumerate(range_n_clusters):
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        
        clusterer2 = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_norm) 
        cluster2_labels = clusterer2.predict(X_norm)
              
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        
        silhouette_avg2.append(silhouette_score(X_norm, cluster2_labels))
        homo2.append(metrics.homogeneity_score(y, cluster2_labels))
        comp2.append(metrics.completeness_score(y, cluster2_labels))
        NMI2.append(normalized_mutual_info_score(y, cluster2_labels))
        AIC.append(clusterer2.aic(X_norm)) 
        BIC.append(clusterer2.bic(X_norm))  
    end = time.perf_counter() 
    print("EM run time: %.1f [s]" % (start-end))    
    plt.plot(range_n_clusters, silhouette_avg2, label="silhouette")
    plt.plot(range_n_clusters, homo2, label="homogeneity")
    plt.plot(range_n_clusters, comp2, label="completeness")
    plt.plot(range_n_clusters, NMI2, label="NMI")
    plt.ylabel('value')
    plt.xlabel('number of cluster')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    
    plt.plot(range_n_clusters, AIC, label="AIC")
    plt.plot(range_n_clusters, BIC, label="BIC")
    plt.ylabel('value')
    plt.xlabel('number of cluster')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()

    #visulization of clusters
    k1 = 4
    plt.figure()
    plt.hist(cluster2_labels, bins=np.arange(0, k1 + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k1))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.show()

#################################################
def run_PCA(X,y,title):
    
    pca = PCA(random_state=5).fit(X) #for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Explained Variance and Eigenvalues: "+ title)
    fig.tight_layout()
    plt.show()
    
    
def run_ICA(X,y,title):
    
    dims = list(np.arange(1,X.shape[1]+1))
    ica = FastICA(random_state=10)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.mean())

    plt.figure()
    plt.title("ICA Kurtosis: "+ title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.show()


import scipy.sparse as sps
from scipy.linalg import pinv
def reconstruction_error(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p@w)@(x.T)).T  # Unproject projected data
    errors = np.square(x-reconstructed)
    return np.nanmean(errors) 
    
def run_SRP(X,y,title):
    
    dims = list(np.arange(1,X.shape[1]+1))   
    tmp1 = defaultdict(dict)
    for i,dim in product(range(3),dims):
        rp = SRP(random_state=5, n_components=dim)
        rp = rp.fit(X)
        tmp1[dim][i] = reconstruction_error(rp, X)
    tmp1 = pd.DataFrame(tmp1).T

    plt.plot(dims,tmp1, 'm-')
    plt.ylabel('error')
    plt.xlabel('number of dimension')
    plt.legend(loc="best")
    plt.title("Random Components for 3 Restarts: "+title)
    plt.show()


def run_FA(X,y,title):
    
    fa = FA(random_state=5)
    fa.fit_transform(X)
    vn = fa.noise_variance_
    print(vn)
    plt.plot(list(range(len(vn))), vn, 'm-')
    plt.xlabel('conponent')
    plt.ylabel('noise variance')
    plt.tick_params('y', colors='m')
    plt.title("FA Noise Variance: "+ title)
    plt.show()

from sklearn.decomposition import TruncatedSVD as SVD
def run_SVD(X,y,title):
    
    dims = list(np.arange(1,X.shape[1]))
    svd = SVD(random_state=10)

    for dim in dims:
        svd.set_params(n_components=dim)
        svd.fit_transform(X)
        ev = svd.explained_variance_ratio_
        cum_var = np.cumsum(ev)

    fig, ax1 = plt.subplots()
    ax1.plot(dims, cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims, ev, 'm-')
    ax2.set_ylabel('Explained Variance Ratio', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("SVD: "+ title)
    fig.tight_layout()
    plt.show()

#part 1(cluster), 2 (dimension reduction)
run_Kmeans(X_norm, y,"car_Kmeans")
run_EM(X_norm, y,"car_EM")
run_ICA(X_norm, y, "car")
run_PCA(X_norm, y, "car")
run_SRP(X_norm, y, "car")
run_FA(X_norm, y, "car")
run_SVD(X_norm, y, "car")

#part 3 (dimension reducted and then cluster)
#################################################
#K means with dimensionality reduction
def run_Kmeans_dr(X_norm,y,title):
    range_n_clusters = [2,3,4,5,6]
    loss_n = []
    loss_pca = []
    loss_ica = []
    loss_srp = []
    loss_svd = []
    silhouette_avg0 = []
    silhouette_avg1 = []
    silhouette_avg2 = []
    silhouette_avg3 = []
    silhouette_avg4 = []
    homo0 = []
    homo1 = []
    homo2 = []
    homo3 = []
    homo4 = []
    comp0 = []
    comp1 = []
    comp2 = []
    comp3 = []
    comp4 = []
    
    for index, n_clusters in enumerate(range_n_clusters):
    
        X_pca = PCA(n_components=3,random_state=5).fit_transform(X_norm)
        X_ica = FastICA(n_components=2,random_state=5).fit_transform(X_norm)
        X_srp = SRP(n_components=4,random_state=5).fit_transform(X_norm)
        X_svd = SVD(n_components=4,random_state=5).fit_transform(X_norm)
        clusterer_n = KMeans(n_clusters=n_clusters, random_state=10).fit(X_norm)
        clusterer_pca = KMeans(n_clusters=n_clusters, random_state=10).fit(X_pca)
        clusterer_ica = KMeans(n_clusters=n_clusters, random_state=10).fit(X_ica)
        clusterer_srp = KMeans(n_clusters=n_clusters, random_state=10).fit(X_srp)
        clusterer_svd = KMeans(n_clusters=n_clusters, random_state=10).fit(X_svd)
        loss_n.append(clusterer_n.inertia_)
        loss_pca.append(clusterer_pca.inertia_)
        loss_ica.append(clusterer_ica.inertia_)    
        loss_srp.append(clusterer_srp.inertia_)
        loss_svd.append(clusterer_svd.inertia_)   
        cluster0_labels = clusterer_n.labels_
        cluster1_labels = clusterer_pca.labels_
        cluster2_labels = clusterer_ica.labels_
        cluster3_labels = clusterer_srp.labels_
        cluster4_labels = clusterer_svd.labels_
        silhouette_avg0.append(silhouette_score(X_norm, cluster1_labels))
        silhouette_avg1.append(silhouette_score(X_pca, cluster1_labels))
        silhouette_avg2.append(silhouette_score(X_ica, cluster1_labels))
        silhouette_avg3.append(silhouette_score(X_srp, cluster1_labels))
        silhouette_avg4.append(silhouette_score(X_svd, cluster1_labels))
        homo0.append(metrics.homogeneity_score(y, cluster0_labels))
        homo1.append(metrics.homogeneity_score(y, cluster1_labels))
        homo2.append(metrics.homogeneity_score(y, cluster2_labels))
        homo3.append(metrics.homogeneity_score(y, cluster3_labels))
        homo4.append(metrics.homogeneity_score(y, cluster4_labels))
        comp0.append(metrics.completeness_score(y, cluster0_labels))
        comp1.append(metrics.completeness_score(y, cluster1_labels))
        comp2.append(metrics.completeness_score(y, cluster2_labels))
        comp3.append(metrics.completeness_score(y, cluster3_labels))
        comp4.append(metrics.completeness_score(y, cluster4_labels))

    
    #SSE
    plt.plot(range_n_clusters, loss_n, label="without rd")
    plt.plot(range_n_clusters, loss_pca, label="pca")
    plt.plot(range_n_clusters, loss_ica, label="ica")
    plt.plot(range_n_clusters, loss_srp, label="srp")
    plt.plot(range_n_clusters, loss_svd, label="svd")
    plt.xlabel('number of cluster')
    plt.title('Kmeams:SSE')
    plt.legend(loc="best")
    plt.show()
    
    #silhouette
    plt.plot(range_n_clusters, silhouette_avg0, label="without rd")
    plt.plot(range_n_clusters, silhouette_avg1, label="pca")
    plt.plot(range_n_clusters, silhouette_avg2, label="ica")
    plt.plot(range_n_clusters, silhouette_avg3, label="srp")
    plt.plot(range_n_clusters, silhouette_avg4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('Kmeams:silhouette')
    plt.legend(loc="best")
    plt.show()
    
    #homogeneity
    plt.plot(range_n_clusters, homo0, label="without rd")
    plt.plot(range_n_clusters, homo1, label="pca")
    plt.plot(range_n_clusters, homo2, label="ica")
    plt.plot(range_n_clusters, homo3, label="srp")
    plt.plot(range_n_clusters, homo4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('Kmeams:homogeneity')
    plt.legend(loc="best")
    plt.show()
    
    #completeness
    plt.plot(range_n_clusters, comp0, label="without rd")
    plt.plot(range_n_clusters, comp1, label="pca")
    plt.plot(range_n_clusters, comp2, label="ica")
    plt.plot(range_n_clusters, comp3, label="srp")
    plt.plot(range_n_clusters, comp4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('Kmeams:completeness')
    plt.legend(loc="best")
    plt.show()


#EM with dimensionality reduction
def run_EM_dr(X_norm,y,title):
    range_n_clusters = [2,3,4,5,6]
    loss_n = []
    loss_pca = []
    loss_ica = []
    loss_srp = []
    loss_svd = []
    silhouette_avg0 = []
    silhouette_avg1 = []
    silhouette_avg2 = []
    silhouette_avg3 = []
    silhouette_avg4 = []
    homo0 = []
    homo1 = []
    homo2 = []
    homo3 = []
    homo4 = []
    comp0 = []
    comp1 = []
    comp2 = []
    comp3 = []
    comp4 = []
    
    for index, n_clusters in enumerate(range_n_clusters):
    
        X_pca = PCA(n_components=3,random_state=5).fit_transform(X_norm)
        X_ica = FastICA(n_components=2,random_state=5).fit_transform(X_norm)
        X_srp = SRP(n_components=4,random_state=5).fit_transform(X_norm)
        X_svd = SVD(n_components=4,random_state=5).fit_transform(X_norm)
        clusterer_n =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_norm)
        clusterer_pca =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_pca)
        clusterer_ica =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_ica)
        clusterer_srp =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_srp)
        clusterer_svd =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_svd)
        loss_n.append(clusterer_n.bic(X_norm))
        loss_pca.append(clusterer_pca.aic(X_pca))
        loss_ica.append(clusterer_ica.aic(X_ica))    
        loss_srp.append(clusterer_srp.aic(X_srp))
        loss_svd.append(clusterer_svd.aic(X_svd))   
        cluster0_labels = clusterer_n.predict(X_norm)
        cluster1_labels = clusterer_pca.predict(X_pca)
        cluster2_labels = clusterer_ica.predict(X_ica)
        cluster3_labels = clusterer_srp.predict(X_srp)
        cluster4_labels = clusterer_svd.predict(X_svd)
        silhouette_avg0.append(silhouette_score(X_norm, cluster1_labels))
        silhouette_avg1.append(silhouette_score(X_pca, cluster1_labels))
        silhouette_avg2.append(silhouette_score(X_ica, cluster1_labels))
        silhouette_avg3.append(silhouette_score(X_srp, cluster1_labels))
        silhouette_avg4.append(silhouette_score(X_svd, cluster1_labels))
        homo0.append(metrics.homogeneity_score(y, cluster0_labels))
        homo1.append(metrics.homogeneity_score(y, cluster1_labels))
        homo2.append(metrics.homogeneity_score(y, cluster2_labels))
        homo3.append(metrics.homogeneity_score(y, cluster3_labels))
        homo4.append(metrics.homogeneity_score(y, cluster4_labels))
        comp0.append(metrics.completeness_score(y, cluster0_labels))
        comp1.append(metrics.completeness_score(y, cluster1_labels))
        comp2.append(metrics.completeness_score(y, cluster2_labels))
        comp3.append(metrics.completeness_score(y, cluster3_labels))
        comp4.append(metrics.completeness_score(y, cluster4_labels))

    
    #BIC
    plt.plot(range_n_clusters, loss_n, label="without rd")
    plt.plot(range_n_clusters, loss_pca, label="pca")
    plt.plot(range_n_clusters, loss_ica, label="ica")
    plt.plot(range_n_clusters, loss_srp, label="srp")
    plt.plot(range_n_clusters, loss_svd, label="svd")
    plt.xlabel('number of cluster')
    plt.title('EM:BIC')
    plt.legend(loc="best")
    plt.show()
    
    #silhouette
    plt.plot(range_n_clusters, silhouette_avg0, label="without rd")
    plt.plot(range_n_clusters, silhouette_avg1, label="pca")
    plt.plot(range_n_clusters, silhouette_avg2, label="ica")
    plt.plot(range_n_clusters, silhouette_avg3, label="srp")
    plt.plot(range_n_clusters, silhouette_avg4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('EM:silhouette')
    plt.legend(loc="best")
    plt.show()
    
    #homogeneity
    plt.plot(range_n_clusters, homo0, label="without rd")
    plt.plot(range_n_clusters, homo1, label="pca")
    plt.plot(range_n_clusters, homo2, label="ica")
    plt.plot(range_n_clusters, homo3, label="srp")
    plt.plot(range_n_clusters, homo4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('EM:homogeneity')
    plt.legend(loc="best")
    plt.show()
    
    #completeness
    plt.plot(range_n_clusters, comp0, label="without rd")
    plt.plot(range_n_clusters, comp1, label="pca")
    plt.plot(range_n_clusters, comp2, label="ica")
    plt.plot(range_n_clusters, comp3, label="srp")
    plt.plot(range_n_clusters, comp4, label="svd")
    plt.xlabel('number of cluster')
    plt.title('EM:completeness')
    plt.legend(loc="best")
    plt.show()


run_Kmeans_dr(X_norm,y,'Kmeans with dimensionality reduction')
run_EM_dr(X_norm,y,'EM with dimensionality reduction')












