# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:49:06 2019

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

data = pd.read_csv("./letter-recognition.csv")
y=data.iloc[:,[0]]
x=data.iloc[:,[i for i in range(1,17)]]

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)

scaler = MinMaxScaler(feature_range=[0,100])
scaler.fit(x)
X_norm = pd.DataFrame(scaler.transform(x))

#################################################
#K means clustering
def run_Kmeans(X_norm,y,title):
    range_n_clusters = [i for i in range(2,26)]
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
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg1.append(silhouette_score(X_norm, cluster1_labels))
        homo1.append(metrics.homogeneity_score(y, cluster1_labels))
        comp1.append(metrics.completeness_score(y, cluster1_labels))
        NMI1.append(normalized_mutual_info_score(y, cluster1_labels))
    
    end = time.perf_counter() 
    print("Keams run time: %.1f [s]" % (start-end))
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
    k1 = 6
    clusterer1 = KMeans(n_clusters=k1, random_state=10).fit(X_norm)
    loss1.append(clusterer1.inertia_)    
    cluster1_labels = clusterer1.labels_
    plt.figure()
    plt.hist(clusterer1.labels_, bins=np.arange(0, k1 + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, k1))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title(title+ ":data distribution")
    plt.show()

#EM
def run_EM(X_norm,y,title):
    range_n_clusters = [i for i in range(2,27)]
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
    start = time.perf_counter() 
    pca = PCA(random_state=5).fit(X) #for all components
    end = time.perf_counter() 
    print("sPCA run time: %.1f [s]" % (start-end))
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
    start = time.perf_counter() 
    dims = list(np.arange(1,X.shape[1]+1))
    ica = FastICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.mean())
        
    end = time.perf_counter() 
    print("sICA run time: %.1f [s]" % (start-end))    
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
    start = time.perf_counter() 
    dims = list(np.arange(1,X.shape[1]+1))   
    tmp1 = defaultdict(dict)
    for i,dim in product(range(3),dims):
        rp = SRP(random_state=5, n_components=dim)
        rp = rp.fit(X)
        tmp1[dim][i] = reconstruction_error(rp, X)
    tmp1 = pd.DataFrame(tmp1).T
    end = time.perf_counter() 
    print("sSRP run time: %.1f [s]" % (start-end))    
    plt.plot(dims,tmp1, 'm-')
    plt.ylabel('error')
    plt.xlabel('number of dimension')
    plt.legend(loc="best")
    plt.title("Random Components for 3 Restarts: "+title)
    plt.show()
    
from sklearn.decomposition import TruncatedSVD as SVD
def run_SVD(X,y,title):
    start = time.perf_counter() 
    dims = list(np.arange(1,X.shape[1]))
    svd = SVD(random_state=10)

    for dim in dims:
        svd.set_params(n_components=dim)
        svd.fit_transform(X)
        ev = svd.explained_variance_ratio_
        cum_var = np.cumsum(ev)
    end = time.perf_counter() 
    print("sSVD run time: %.1f [s]" % (start-end))   
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
run_Kmeans(X_norm, y,"letter_Kmeans")
run_EM(X_norm, y,"letter_EM")
run_ICA(X_norm, y, "letter")
run_PCA(X_norm, y, "letter")
run_SRP(X_norm, y, "letter")
run_SVD(X_norm, y, "letter")

#part 3 (dimension reducted and then cluster)
#################################################
#K means with dimensionality reduction
def run_Kmeans_dr(X_norm,y,title):
    range_n_clusters = [i for i in range(2,26)]
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
    
        X_pca = PCA(n_components=4,random_state=5).fit_transform(X_norm)
        X_ica = FastICA(n_components=7,random_state=5).fit_transform(X_norm)
        X_srp = SRP(n_components=4,random_state=5).fit_transform(X_norm)
        X_svd = SVD(n_components=5,random_state=5).fit_transform(X_norm)
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
    plt.title(title+':SSE')
    plt.legend(loc="best")
    plt.show()
    
    #silhouette
    plt.plot(range_n_clusters, silhouette_avg0, label="without rd")
    plt.plot(range_n_clusters, silhouette_avg1, label="pca")
    plt.plot(range_n_clusters, silhouette_avg2, label="ica")
    plt.plot(range_n_clusters, silhouette_avg3, label="srp")
    plt.plot(range_n_clusters, silhouette_avg4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':silhouette')
    plt.legend(loc="best")
    plt.show()
    
    #homogeneity
    plt.plot(range_n_clusters, homo0, label="without rd")
    plt.plot(range_n_clusters, homo1, label="pca")
    plt.plot(range_n_clusters, homo2, label="ica")
    plt.plot(range_n_clusters, homo3, label="srp")
    plt.plot(range_n_clusters, homo4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':homogeneity')
    plt.legend(loc="best")
    plt.show()
    
    #completeness
    plt.plot(range_n_clusters, comp0, label="without rd")
    plt.plot(range_n_clusters, comp1, label="pca")
    plt.plot(range_n_clusters, comp2, label="ica")
    plt.plot(range_n_clusters, comp3, label="srp")
    plt.plot(range_n_clusters, comp4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':completeness')
    plt.legend(loc="best")
    plt.show()


#EM with dimensionality reduction
def run_EM_dr(X_norm,y,title):
    range_n_clusters = [i for i in range(2,26)]
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
    
        X_pca = PCA(n_components=4,random_state=5).fit_transform(X_norm)
        X_ica = FastICA(n_components=7,random_state=5).fit_transform(X_norm)
        X_srp = SRP(n_components=5,random_state=5).fit_transform(X_norm)
        X_svd = SVD(n_components=6,random_state=5).fit_transform(X_norm)
        clusterer_n =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_norm)
        clusterer_pca =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_pca)
        clusterer_ica =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_ica)
        clusterer_srp =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_srp)
        clusterer_svd =  GaussianMixture(n_components=n_clusters, random_state=10).fit(X_svd)
        loss_n.append(clusterer_n.bic(X_norm))
        loss_pca.append(clusterer_pca.bic(X_pca))
        loss_ica.append(clusterer_ica.bic(X_ica))    
        loss_srp.append(clusterer_srp.bic(X_srp))
        loss_svd.append(clusterer_svd.bic(X_svd))   
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
    plt.xlabel(title+':number of cluster')
    plt.title(title+':BIC')
    plt.legend(loc="best")
    plt.show()
    
    #silhouette
    plt.plot(range_n_clusters, silhouette_avg0, label="without rd")
    plt.plot(range_n_clusters, silhouette_avg1, label="pca")
    plt.plot(range_n_clusters, silhouette_avg2, label="ica")
    plt.plot(range_n_clusters, silhouette_avg3, label="srp")
    plt.plot(range_n_clusters, silhouette_avg4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':silhouette')
    plt.legend(loc="best")
    plt.show()
    
    #homogeneity
    plt.plot(range_n_clusters, homo0, label="without rd")
    plt.plot(range_n_clusters, homo1, label="pca")
    plt.plot(range_n_clusters, homo2, label="ica")
    plt.plot(range_n_clusters, homo3, label="srp")
    plt.plot(range_n_clusters, homo4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':homogeneity')
    plt.legend(loc="best")
    plt.show()
    
    #completeness
    plt.plot(range_n_clusters, comp0, label="without rd")
    plt.plot(range_n_clusters, comp1, label="pca")
    plt.plot(range_n_clusters, comp2, label="ica")
    plt.plot(range_n_clusters, comp3, label="srp")
    plt.plot(range_n_clusters, comp4, label="svd")
    plt.xlabel('number of cluster')
    plt.title(title+':completeness')
    plt.legend(loc="best")
    plt.show()


run_Kmeans_dr(X_norm,y,'Kmeans')
run_EM_dr(X_norm,y,'EM')

#part 4 (NN with dimension reduction)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def learning_curve_nnrd(r, X_norm,y):
    train_pca = []
    train_ica = []
    train_rca = []
    train_svd = []
    clf = MLPClassifier(hidden_layer_sizes=(26))
    start = time.perf_counter() 
    for n in range(2, X_norm.shape[1]):
        X_pca = PCA(n_components=n,random_state=5).fit_transform(X_norm)
        nn_pca = clf.fit(X_pca, y)
        p_pca = nn_pca.predict(X_pca)       
        train_pca.append(accuracy_score(y, p_pca))
    end = time.perf_counter() 
    print("pca run time: %.1f [s]" % (start-end))
    
    start = time.perf_counter() 
    for n in range(2, X_norm.shape[1]):
        X_svd = SVD(n_components=n,random_state=5).fit_transform(X_norm)
        nn_svd = clf.fit(X_svd, y)
        p_svd = nn_svd.predict(X_svd)
        train_svd.append(accuracy_score(y, p_svd))
    end = time.perf_counter() 
    print("svd run time: %.1f [s]" % (start-end))
    
    start = time.perf_counter()     
    for n in range(2, X_norm.shape[1]):
        X_ica = FastICA(n_components=n,random_state=5).fit_transform(X_norm)
        nn_ica = clf.fit(X_ica, y)
        p_ica = nn_ica.predict(X_ica)
        train_ica.append(accuracy_score(y, p_ica))
    end = time.perf_counter() 
    print("ica run time: %.1f [s]" % (start-end))
    
    start = time.perf_counter()   
    for n in range(2, X_norm.shape[1]):
        X_srp = SRP(n_components=n,random_state=5).fit_transform(X_norm)
        nn_srp = clf.fit(X_srp, y)
        p_srp = nn_srp.predict(X_srp)
        train_rca.append(accuracy_score(y, p_srp))
    end = time.perf_counter() 
    print("srp run time: %.1f [s]" % (start-end))
    
    plt.plot(list(range(2, X_norm.shape[1])),train_pca, label="pca")
    plt.plot(list(range(2, X_norm.shape[1])),train_ica, label="ica")
    plt.plot(list(range(2, X_norm.shape[1])),train_rca, label="srp")
    plt.plot(list(range(2, X_norm.shape[1])),train_svd, label="svd")
    plt.ylabel('accuracy')
    plt.xlabel('number of components')
    plt.legend(loc="best")
    plt.show()

learning_curve_nnrd(5, X_norm, y)

#part 5 (NN with cluster after dimension reduction)
from sklearn.model_selection import train_test_split

def plot_nnrdc_curve(title,x,y1,y2,y3):
    train1=[]
    train2=[]
    train3=[]
    test1=[]
    test2=[]
    test3=[]
    fraction = [i/10 for i in range(1,10)]
    for i in range(1,10):
        clf = MLPClassifier(hidden_layer_sizes=(26))   # change the hidden_layer_sizes = (26) or (16, 8) for parameter tuning
        X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y1, random_state=0, test_size=1-i*10/100)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x, y2, random_state=0, test_size=1-i*10/100)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(x, y3, random_state=0, test_size=1-i*10/100)
        
        clf.fit(X_train1, y_train1)    
        train_predict1 = clf.predict(X_train1)
        train1.append(accuracy_score(y_train1, train_predict1))
        test_predict1 = clf.predict(X_test1)
        test1.append(accuracy_score(y_test1, test_predict1))
        
        clf.fit(X_train2, y_train2)
        train_predict2 = clf.predict(X_train2)
        train2.append(accuracy_score(y_train2, train_predict2))
        test_predict2 = clf.predict(X_test2)
        test2.append(accuracy_score(y_test2, test_predict2))
        
        clf.fit(X_train3, y_train3)
        train_predict3 = clf.predict(X_train3)
        train3.append(accuracy_score(y_train3, train_predict3))
        test_predict3 = clf.predict(X_test3)
        test3.append(accuracy_score(y_test3, test_predict3))

    plt.plot(fraction,train1, label="training:without clustering", linestyle='-', color='b')
    plt.plot(fraction,train2, label="training:KMemas", linestyle='-', color='g')
    plt.plot(fraction,train3, label="training:EM", linestyle='-', color='r')
    plt.plot(fraction,test1, label="test:without clustering", linestyle='--', color='b')
    plt.plot(fraction,test2, label="test:KMemas", linestyle='--', color='g')
    plt.plot(fraction,test3, label="test:EM", linestyle='--', color='r')
    plt.ylabel('accuracy')
    plt.xlabel('fractional trainning data size')
    plt.legend(loc="best")
    plt.title('NN with '+title)
    plt.show()


def learning_curve_nnc(r, X_norm,y):
    X_pca = PCA(n_components=6,random_state=5).fit_transform(X_norm)  
    KM_pca = KMeans(n_clusters=10, random_state=10).fit(X_pca)
    EM_pca =  GaussianMixture(n_components=10, random_state=10).fit(X_pca)
    KM_pca_labels = KM_pca.labels_
    EM_pca_labels = EM_pca.predict(X_pca)
    
    X_ica = FastICA(n_components=6,random_state=5).fit_transform(X_norm)  
    KM_ica = KMeans(n_clusters=10, random_state=10).fit(X_ica)
    EM_ica =  GaussianMixture(n_components=10, random_state=10).fit(X_ica)
    KM_ica_labels = KM_ica.labels_
    EM_ica_labels = EM_ica.predict(X_ica)   
    
    X_srp = SRP(n_components=6,random_state=5).fit_transform(X_norm)  
    KM_srp = KMeans(n_clusters=10, random_state=10).fit(X_srp)
    EM_srp =  GaussianMixture(n_components=10, random_state=10).fit(X_srp)
    KM_srp_labels = KM_srp.labels_
    EM_srp_labels = EM_srp.predict(X_srp)
    
    X_svd = SVD(n_components=6,random_state=5).fit_transform(X_norm)  
    KM_svd = KMeans(n_clusters=10, random_state=10).fit(X_svd)
    EM_svd =  GaussianMixture(n_components=10, random_state=10).fit(X_svd)
    KM_svd_labels = KM_svd.labels_
    EM_svd_labels = EM_svd.predict(X_svd)  
    
    plot_nnrdc_curve('PCA',X_pca,y,KM_pca_labels,EM_pca_labels)
    plot_nnrdc_curve('ICA',X_ica,y,KM_ica_labels,EM_ica_labels)
    plot_nnrdc_curve('SRP',X_srp,y,KM_srp_labels,EM_srp_labels)
    plot_nnrdc_curve('SVD',X_svd,y,KM_svd_labels,EM_svd_labels)
    


learning_curve_nnc(5, X_norm, y)




