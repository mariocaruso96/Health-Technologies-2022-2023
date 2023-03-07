# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:46:44 2023

#Use this code for the connectome analysis.

@author: mik16
"""

#%% Librerie utili

from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr, ttest_ind, kstest, mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne as mne
import os
import networkx as nx
import statsmodels.api as sm
import seaborn as sns


#%% Load Dataframes

def load_dataframe(path, start_sub, end_sub, samp_freq):
    
    time_step = 1/samp_freq
    d_rest = {}
    d_snow_pt1 = {}
    d_snow_pt2 = {}
    d_urban = {}
    
    for n_sub in range(start_sub, end_sub + 1):
        rest = mne.io.read_raw_eeglab(path + str(n_sub) + "/rest" + str(n_sub) + ".set")
        rest = pd.DataFrame(rest[0: 16][0])
        rest = rest.transpose()
        index_can = np.arange(0, rest.shape[0], 1)
        #tempi effettivi
        time = np.round((index_can * time_step), 3)
        #indicizziamo il dataframe
        rest['time'] = pd.Series(time)
        rest = rest.set_index('time')
        d_rest["rest{0}".format(n_sub)] = rest
        
        snow_pt1 = mne.io.read_raw_eeglab(path + str(n_sub) + "/snow" + str(n_sub) + "_pt1.set")
        snow_pt1 = pd.DataFrame(snow_pt1[0: 16][0])
        snow_pt1 = snow_pt1.transpose()
        index_can = np.arange(0, snow_pt1.shape[0], 1)
        #tempi effettivi
        time = np.round((index_can * time_step), 3)
        #indicizziamo il dataframe
        snow_pt1['time'] = pd.Series(time)
        snow_pt1 = snow_pt1.set_index('time')
        d_snow_pt1["snow{0}".format(n_sub) + str(1)] = snow_pt1
        
        snow_pt2 = mne.io.read_raw_eeglab(path + str(n_sub) + "/snow" + str(n_sub) + "_pt2.set")
        snow_pt2 = pd.DataFrame(snow_pt2[0: 16][0])
        snow_pt2 = snow_pt2.transpose()
        index_can = np.arange(0, snow_pt2.shape[0], 1)
        #tempi effettivi
        time = np.round((index_can * time_step), 3)
        #indicizziamo il dataframe
        snow_pt2['time'] = pd.Series(time)
        snow_pt2 = snow_pt2.set_index('time')
        d_snow_pt2["snow{0}".format(n_sub)+ str(2)] = snow_pt2

        urban = mne.io.read_raw_eeglab(path + str(n_sub) + "/urban" + str(n_sub) + ".set")
        urban = pd.DataFrame(urban[0: 16][0])
        urban = urban.transpose()
        index_can = np.arange(0, urban.shape[0], 1)
        #tempi effettivi
        time = np.round((index_can * time_step), 3)
        #indicizziamo il dataframe
        urban['time'] = pd.Series(time)
        urban = urban.set_index('time')
        d_urban["urban{0}".format(n_sub)] = urban
        
        #indicizzazione
        
        #serve per indicizzare le righe del dataframe come tempi effettivi
        
    return d_rest, d_snow_pt1, d_snow_pt2, d_urban
        

#%% Import Dataframes

path = "C:/Users/carus/Desktop/Health Technologies/Acquisizioni/Clean/Sub"
samp_freq = 125
d_rest, d_snow_pt1, d_snow_pt2, d_urban = load_dataframe(path, 1, 6, samp_freq)

del path, samp_freq


#%% Generate Connectome

#Creazione del Connettoma
def generate_connectome(d, threshold = None, delete_under_threshold = False):
    
    """
    Parametri:
    - x = oggetto DataFrame di pandas
    - num_channels = intero, numero di canali disponibili
    - threshold = soglia nel calcolo della correlazione di Pearson al di sotto della quale
                  porre tutto a 0 se delete_under_threshold = True
    
    Uscita:
    - G = grafo, oggetto di networkx
    - Ad = numpy array, matrice di adiacenza del grafo G
    
    Spiegazione:
    La funzione crea una matrice di adiacenza Ad di un grafo di num_channels nodi che sia, almeno
    inizialmente, completo e non diretto. Ogni nodo rappresenta un elettrodo dell'elettrencefalografo.
    La matrice Ad deve essere una matrice quadrata di num_channels x num_channels dove l'elemento
    Ad(i,j) è il coefficiente di correlazione di Pearson calcolato per i canali i e j: esso diventa il
    peso del link tra nodo i e nodo j del grafo.
    
    """
    d_G = {}
    d_Ad= {}
    
    for key in d:
        df = d[key]
        s = (df.shape[1], df.shape[1])
        Ad = np.zeros(s)
        
        for i in range(0, Ad.shape[0]):
            for j in range(0, Ad.shape[1]):
                pcorr, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
                
                if delete_under_threshold == False:
                    Ad[i, j] = abs(pcorr)
                    Ad[i, i] = 0
                
                if delete_under_threshold == True:
                    if abs(pcorr) > threshold:
                        Ad[i, j] = abs(pcorr)
                        Ad[i, i] = 0
                    else:
                        Ad[i, j] = 0
                        Ad[i, i] = 0
                        
        d_Ad[str(key)] = Ad          
        G = nx.from_numpy_array(Ad)
        d_G[str(key)] = G     
        
    return d_Ad, d_G
        

#%% Create Graphs and Adj Matrices

d_Ad_rest, d_G_rest = generate_connectome(d_rest, 0.2, True)
d_Ad_snow_pt1, d_G_snow_pt1 = generate_connectome(d_snow_pt1, 0.2, True)
d_Ad_snow_pt2, d_G_snow_pt2 = generate_connectome(d_snow_pt2, 0.2, True)
d_Ad_urban, d_G_urban = generate_connectome(d_urban, 0.2, True)

#del d_rest, d_snow_pt1, d_snow_pt2, d_urban

#%% Boxplot Pearson correlation coefficients

boxprops = dict(color="black", linewidth = 1)
medianprops = dict(color="red", linewidth = 1)
labels = ['rest', 'snow1', 'snow2', 'urban']


list_1 = [d_Ad_rest['rest1'].flatten(), d_Ad_snow_pt1['snow11'].flatten(), d_Ad_snow_pt2['snow12'].flatten(), d_Ad_urban['urban1'].flatten()]
list_2 = [d_Ad_rest['rest2'].flatten(), d_Ad_snow_pt1['snow21'].flatten(), d_Ad_snow_pt2['snow22'].flatten(), d_Ad_urban['urban2'].flatten()]
list_3 = [d_Ad_rest['rest3'].flatten(), d_Ad_snow_pt1['snow31'].flatten(), d_Ad_snow_pt2['snow32'].flatten(), d_Ad_urban['urban3'].flatten()]
list_4 = [d_Ad_rest['rest4'].flatten(), d_Ad_snow_pt1['snow41'].flatten(), d_Ad_snow_pt2['snow42'].flatten(), d_Ad_urban['urban4'].flatten()]
list_5 = [d_Ad_rest['rest5'].flatten(), d_Ad_snow_pt1['snow51'].flatten(), d_Ad_snow_pt2['snow52'].flatten(), d_Ad_urban['urban5'].flatten()]
list_6 = [d_Ad_rest['rest6'].flatten(), d_Ad_snow_pt1['snow61'].flatten(), d_Ad_snow_pt2['snow62'].flatten(), d_Ad_urban['urban6'].flatten()]

savepath = "C:/Users/carus/Desktop/Health Technologies/Immagini/Boxplot Pearson coefficients"

def pearson_boxplot(list_n, title):
    bplot = plt.boxplot(list_n, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title("Pearson's correlation coefficients boxplot " + title)


pearson_boxplot(list_1, "Subject 1")
plt.savefig(savepath + "/" + "Subject1.png", dpi = 300)
plt.show()

pearson_boxplot(list_2, "Subject 2")
plt.savefig(savepath + "/" + "Subject2.png", dpi = 300)
plt.show()

pearson_boxplot(list_3, "Subject 3")
plt.savefig(savepath + "/" + "Subject3.png", dpi = 300)
plt.show()

pearson_boxplot(list_4, "Subject 4")
plt.savefig(savepath + "/" + "Subject4.png", dpi = 300)
plt.show()

pearson_boxplot(list_5, "Subject 5")
plt.savefig(savepath + "/" + "Subject5.png", dpi = 300)
plt.show()

pearson_boxplot(list_6, "Subject 6")
plt.savefig(savepath + "/" + "Subject6.png", dpi = 300)
plt.show()


#%% Metrics boxplots 

boxprops = dict(color="black", linewidth = 1)
medianprops = dict(color="red", linewidth = 1)
labels = ['rest', 'snow1', 'snow2', 'urban']

def deg_box(sub_graphs, num):
    deg_list = []
    for i in range(0, len(sub_graphs)):
        deg_list.append(list(dict(nx.degree(sub_graphs[i])).values()))
        
    bplot = plt.boxplot(deg_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Degrees Boxplots Subject {num}")
    #plt.show()
    return deg_list
    

def bet_box(sub_graphs, num):
    bet_list = []
    for i in range(0, len(sub_graphs)):
        bet_list.append(list(dict(nx.betweenness_centrality(sub_graphs[i])).values()))
        
    bplot = plt.boxplot(bet_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Betweenness Boxplots Subject {num}")
    #plt.show()
    return bet_list
  
    
def edge_bet_box(sub_graphs, num):
    edge_bet_list = []
    for i in range(0, len(sub_graphs)):
        edge_bet_list.append(list(dict(nx.edge_betweenness_centrality(sub_graphs[i])).values()))

    bplot = plt.boxplot(edge_bet_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Edge Betweenness Boxplots Subject {num}")
    #plt.show()
    return edge_bet_list


def close_box(sub_graphs, num):
    close_list = []
    for i in range(0, len(sub_graphs)):
        close_list.append(list(dict(nx.closeness_centrality(sub_graphs[i])).values()))
        
    bplot = plt.boxplot(close_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Closeness Boxplots Subject {num}")
    #plt.show()
    return close_list


def info_box(sub_graphs, sub_ads, num):
    info_list = []
    for i in range(0, len(sub_graphs)):
        aux_mat = sub_ads[i]
        aux = sorted(nx.connected_components(sub_graphs[i]), key = len, reverse=True)[0]
        aux_nodes = np.array(list(aux))
        aux_mat = aux_mat[aux_nodes[:, None], aux_nodes]
        aux_G = nx.from_numpy_array(aux_mat)
        info_list.append(list(dict(nx.information_centrality(aux_G)).values()))
    
    bplot = plt.boxplot(info_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Information Boxplots Subject {num}")
    #plt.show()
    return info_list
    
     
def eigen_box(sub_graphs, num):
    eigen_list = []
    for i in range(0, len(sub_graphs)):
        eigen_list.append(list(dict(nx.eigenvector_centrality(sub_graphs[i])).values()))

    bplot = plt.boxplot(eigen_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Eigenvector Centrality Boxplots Subject {num}")
    #plt.show()
    return eigen_list
    

def clust_box(sub_graphs, num):
    clust_list = []
    for i in range(0, len(sub_graphs)):
        clust_list.append(list(dict(nx.clustering(sub_graphs[i])).values()))
        

    bplot = plt.boxplot(clust_list, boxprops = boxprops, medianprops = medianprops, labels = labels, patch_artist = True)
    colors = ["blue", "green", "lightgreen", "orange"]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.title(f"Clustering Coefficients Boxplots Subject {num}")
    #plt.show()
    return clust_list


#%% Metrics Boxplots Generation

savepath = "D:/OneDrive - Università degli Studi di Bari/Scuola/Università/Magistrale/Health Technologies/Final_Project/Images/Sub" 
 
d_deg = {}
d_bet = {}
d_edge_bet = {}
d_close = {}
d_info = {}
d_eigen = {}
d_clust = {}

for key_rest, key_snow1, key_snow2, key_urban in zip(d_Ad_rest, d_Ad_snow_pt1, d_Ad_snow_pt2, d_Ad_urban):
    
    sub_ads = [d_Ad_rest[key_rest], d_Ad_snow_pt1[key_snow1], d_Ad_snow_pt2[key_snow2], d_Ad_urban[key_urban]]
    sub_graphs = [d_G_rest[key_rest], d_G_snow_pt1[key_snow1], d_G_snow_pt2[key_snow2], d_G_urban[key_urban]]
    n_sub = key_rest[-1]
    
    d_deg["Deg_Sub" + str(n_sub)] = deg_box(sub_graphs, n_sub)
    deg_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Degrees Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_bet["Bet_Sub" + str(n_sub)] = bet_box(sub_graphs, n_sub)
    bet_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Betweenness Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_edge_bet["Edge_Bet_Sub" + str(n_sub)] = edge_bet_box(sub_graphs, n_sub)
    edge_bet_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Edge Betweeneess Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_close["Close_Sub" + str(n_sub)] = close_box(sub_graphs, n_sub)
    close_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Closeness Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_info["Info_Sub" + str(n_sub)] = info_box(sub_graphs, sub_ads, n_sub)
    info_box(sub_graphs, sub_ads, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Information Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_eigen["Eigen_Sub" + str(n_sub)] = eigen_box(sub_graphs, n_sub)
    eigen_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Eigenvector Centrality Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    d_clust["Clust_Sub" + str(n_sub)] = clust_box(sub_graphs, n_sub)
    clust_box(sub_graphs, n_sub)
    # plt.savefig(savepath + str(n_sub) + "/Metrics/" + "Clustering Coefficients Boxplot Subject {}.png".format(n_sub), dpi = 300)
    plt.show()
    
del key_rest, key_snow1, key_snow2, key_urban, n_sub, sub_ads, sub_graphs
del d_Ad_rest, d_Ad_snow_pt1, d_Ad_snow_pt2, d_Ad_urban
del d_G_rest, d_G_snow_pt1, d_G_snow_pt2, d_G_urban
del boxprops, labels, medianprops, savepath

#%% Padding Metrics With Mean
    
# def mode_array(array):
    
#     array = np.array(array)
#     vals,counts = np.unique(array, return_counts=True)
#     mode = vals[np.argmax(counts)]
    
#     return mode

def pad_array(d):
    
    for key in d:
        for i in range(0, len(d[str(key)])):
            if len(d[str(key)][i]) != 16:
                size = 16 - len(d[str(key)][i])
                padding = [np.mean(d[str(key)][i])]*size
                d[str(key)][i] = d[str(key)][i] + padding
    return d

d_deg = pad_array(d_deg)
d_bet = pad_array(d_bet)
d_edge_bet = pad_array(d_edge_bet)
d_close = pad_array(d_close)
d_info = pad_array(d_info)
d_eigen = pad_array(d_eigen)
d_clust = pad_array(d_clust)


#%% Dataframe Intersection and Statistal Tests

def df_intersect(df1, df2):
    
    comparisons = ['REST-SNOW1', 'REST-SNOW2', 'REST-URBAN', 'SNOW1-SNOW2','SNOW1-URBAN', 'SNOW2-URBAN']
    df_intersection = pd.DataFrame(np.zeros((df1.shape[0],df1.shape[1])), columns = comparisons )
    for i in range(0, df1.shape[0]):
        for j in range(0, df1.shape[1]):
            if df1.iloc[i,j] != df2.iloc[i,j]:
                df_intersection.iloc[i,j] = 0
            else:
                df_intersection.iloc[i,j] = df1.iloc[i,j]
                
    return df_intersection

def test_stat(d_metric):
    
    p_values_KS = []
    p_values_MW = []

    k = 1
    for key in d_metric:
        p_values_sub_KS = []
        p_values_sub_MW = []
        for i in range(0, len(d_metric[str(key)]) - 1):
            for j in range(i + 1, len(d_metric[key])):
                stat_KS, p_value_KS = kstest(d_metric[str(key)][i], d_metric[str(key)][j])
                stat_MW, p_value_MW = mannwhitneyu(d_metric[str(key)][i], d_metric[str(key)][j])
                p_values_sub_KS.append(p_value_KS)
                p_values_sub_MW.append(p_value_MW)
                print(f" Kolmogorov-Smirnov Test: statistic={stat_KS:.4f}, p-value={p_value_KS:.4f}")
                print(f" Mann–Whitney U Test: statistic={stat_MW:.4f}, p-value={p_value_MW:.4f}")
                print(k)
                k+=1
        p_values_KS.append(p_values_sub_KS)
        p_values_MW.append(p_values_sub_MW)
        
    comparisons = ['REST-SNOW1', 'REST-SNOW2', 'REST-URBAN', 'SNOW1-SNOW2','SNOW1-URBAN', 'SNOW2-URBAN']
    df_KS = pd.DataFrame(p_values_KS, columns = comparisons)
    df_MW = pd.DataFrame(p_values_MW, columns = comparisons)

    #se il p_value < 0.05 rigetto l'ipotesi nulla ovvero che le due distribuzioni coincidono
    #e accetto quella alternativa ovvero che i dati provengono da distribuzioni diverse
    df_KS = (df_KS < 0.05).astype(int)
    df_MW = (df_MW < 0.05).astype(int)
    df_intersection = df_intersect(df_KS, df_MW)
    
    return df_KS, df_MW, df_intersection


#%% p-values Storage
    
df_bet_KS, df_bet_MW, df_bet_int = test_stat(d_bet)
df_close_KS, df_close_MW, df_close_int = test_stat(d_close)
df_clust_KS, df_clust_MW, df_clust_int = test_stat(d_clust)
df_deg_KS, df_deg_MW, df_deg_int = test_stat(d_deg)
df_egde_bet_KS, df_edge_bet_MW, df_edge_bet_int = test_stat(d_edge_bet)
df_eigen_KS, df_eigen_MW, df_eigen_int = test_stat(d_eigen)
df_info_KS, df_info_MW, df_info_int = test_stat(d_info)

dfs_KS = [df_bet_KS, df_close_KS, df_clust_KS, df_deg_KS, df_egde_bet_KS, df_eigen_KS, df_info_KS]
dfs_MW = [df_bet_MW, df_close_MW, df_clust_MW, df_deg_MW, df_edge_bet_MW, df_eigen_MW, df_info_MW]
dfs_int = [df_bet_int, df_close_int, df_clust_int, df_deg_int, df_edge_bet_int, df_eigen_int, df_info_int]

del d_bet, d_close, d_clust, d_deg, d_edge_bet, d_eigen, d_info
del df_bet_KS, df_bet_MW, df_bet_int
del df_close_KS, df_close_MW, df_close_int
del df_clust_KS, df_clust_MW, df_clust_int
del df_deg_KS, df_deg_MW, df_deg_int
del df_egde_bet_KS, df_edge_bet_MW, df_edge_bet_int
del df_eigen_KS, df_eigen_MW, df_eigen_int
del df_info_KS, df_info_MW, df_info_int


#%% Heatmap for the p-values

def create_heatmap(df, title_test, title_metrics, cmap):
    sns.heatmap(df.T, annot = True, cmap = cmap, linewidths = 5, cbar = False)
    plt.xticks(size = 12)
    plt.yticks(size = 12, rotation = 0)
    plt.title(title_test + " Test" + "\nMetric: " + title_metrics, size = 16)

#path_KS = "C:/Users/carus/Desktop/Health Technologies/Immagini/Tabelle di verità/KS"
# path_MW = "C:/Users/carus/Desktop/Health Technologies/Immagini/Tabelle di verità/MN"
#path_KSMW = "C:/Users/carus/Desktop/Health Technologies/Immagini/Tabelle di verità/Consensus KS MN"


tables_titles = ["Beetweenness", "Closeness", "Clustering Coefficient", "Degree", "Edge Betweenness", "Eigenvector Centrality", "Information"]

for i, el in enumerate(dfs_KS):
    #KS
    create_heatmap(el, "Kolmogorov-Smirnov", tables_titles[i], "Blues")
    #plt.savefig(path_KS + "/" + tables_titles[i] + ".png", dpi = 300, bbox_inches = 'tight')
    plt.show()
    #MW
    create_heatmap(dfs_MW[i], "Mann-Whitney", tables_titles[i], "Reds")
    # plt.savefig(path_MW + "/" + tables_titles[i] + ".png", dpi = 300, bbox_inches = 'tight')
    plt.show()
    #KSMW
    create_heatmap(dfs_int[i], "KS - MW Consensus", tables_titles[i], "Greens")
    #plt.savefig(path_KSMW + "/" + tables_titles[i] + ".png", dpi = 300, bbox_inches = 'tight')
    plt.show()

