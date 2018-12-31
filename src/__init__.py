from OPTICS.optics import *
from feat_extraction.feat_extraction import *
from visualizations.visualizations import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import warnings
import pickle
from itertools import combinations
warnings.filterwarnings('ignore')
import os
import datetime as dt

now = dt.datetime.now()
id = str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_Output'
data_loc = "C:/Users/csinclair/Desktop/Misc/DS/DataMining/D_BOM/Sample_data/"
image_loc = "C:/Users/csinclair/Desktop/Misc/DS/DataMining/D_BOM/reports/output/" + id + '/'
os.mkdir(image_loc)
print('output location at {}'.format(image_loc))
dataset = 'STEGS.txt'#input("which file would you like to use? ")

delim = '|'#input("what is the delimiter? ")

print("Loading {} dataset ".format(dataset))
data = pd.read_csv(data_loc + dataset, delimiter = delim)

data = DATA(data).clean()

data.drop_duplicates(inplace = True)

skudata = SKU(data)

min_of = 2

skuscores = pd.DataFrame(columns = ['Name', 'Perc_noise', 'Numclusters','Outliers' , 'DBCV'])

sku_label_dict = {}
sku_pca_dict = {}
outliers = []
iterator = 0

figrows = 2
figcols = 3
#fig, axarray = plt.subplots(figrows, figcols, figsize=(figcols * 7, figrows * 7), subplot_kw={'projection': '3d'})
print("Attempting SKU Stratification")
sku = SKU(data)
transformations = ['UOH',  'OCA', 'Seasonality', 'Type 1']#, 'Facility']
trancombos = [x for x in combinations(transformations, 2)] + [x for x in combinations(transformations, 3)] +  [x for x in combinations(transformations, 4)]  + ['custom']
print(trancombos)
weighted = {0: ', ', 1: ', Weighted'}
store_weights = 'Yes'
#SKU Transformation
skuclust = {}


textfile = open(image_loc + "output.md", "w")
textfile.write("# Da Bom Text Output")
textfile.write("\n <br> ")


for w in [0,1]:
    for tr in trancombos:
        iterator += 1
        print(tr)
        if tr == 'custom':
            data = sku.custom(w)
        else:
            data = sku.scale_base(w, tr)
        data = remove_dups(sku, data)
        numcomps = data.shape[1]
        optics = OPTICS(min_cluster_size=.03)
        optics.fit(data)
        pcacolumns = ['pca' + str(i) for i in np.arange(0, data.shape[1])]
        pc = pd.DataFrame( data, columns = pcacolumns)
        score = {}
        score['Perc_noise'] = len(np.where(optics.labels_ == -1)[0]) / optics.labels_.shape[0]
        score['Numclusters']  = len(np.unique(optics.labels_)) - 1
        score['Name'] = sku.transformation
        score['DBCV'] = optics.validity
        pc['cluster'] = optics.labels_
        outlier_factor = optics.outlier_factor_
        pc['of'] = outlier_factor
        reachability = optics.reachability_
        pc['reach'] = reachability
        outliers = np.where(outlier_factor > min_of)
        outliers = list(outliers[0].ravel())
        pc['cluster'][outliers] = -2
        pc.set_index(sku.indices, 'SKU',  inplace = True)
        score['Outliers'] = len(outliers)
        score['ID'] = str(iterator)
        score = pd.DataFrame(score, index = ['Name'])
        score['columns'] = [sku.columns]
        score['num_components'] = pc.shape[1]
        skuscores = skuscores.append(score)
        skuclust[str(iterator)] = pc
        print(score)

        #if iterator > figrows * figcols:
        #    graph_it = iterator - 6
        #else:
        #    graph_it = iterator
        #row, col = math.floor((graph_it-1) / figcols),(graph_it - 1) % figcols
        #plot_4D(pc,  'SKU {} transformation'.format(sku.transformation), axarray[row, col] )
        """if iterator == (figrows * figcols):
            plt.suptitle("{} iterations of SKU transformations".format(iterator //6), fontsize = 30)
            plt.savefig(image_loc + str(iterator) + "Transformations.png")
            #textfile.write('![{} iterations Transformations]({})'.format(iterator //6, image_loc + str(iterator) + "Transformations.png"))
            #textfile.write('\n <br>')
            plt.clf()
            plt.cla()
            plt.close()
            #plt.show()
            fig, axarray = plt.subplots(figrows, figcols, figsize=(figcols * 7, figrows * 7), subplot_kw={'projection': '3d'})"""


skuscores.sort_values(by = 'DBCV', ascending = False, inplace = True)
#plt.suptitle("{} iterations of SKU transformations".format(iterator), fontsize = 30)
#plt.savefig(image_loc+ str(iterator) + "Transformations.png")
#textfile.write('![{} iterations Transformations]({})'.format(iterator, image_loc + str(iterator) + "Transformations.png"))
#textfile.write('\n <br>')
#plt.clf()
#plt.cla()
#plt.close()
skuscores.set_index('ID', inplace = True)



#print_table(image_loc, skuscores[['Name', 'Perc_noise', 'Numclusters','Outliers' , 'DBCV']][:10])
outdf = sku.savedf
featureweightdict = pd.DataFrame(sku.featureweights)
print(skuscores.drop(columns = 'columns'))
#print(sku.featureweights.keys())
print("")

#plt.show()



sku.savedf.to_csv(data_loc + dataset + 'SKUFinalModel.csv')
choice = '0'
while choice != '-1':
    choice = '0' #input("Which model would you like to output?(0 for all, -1 for finished)")
    if choice == '0':
        for i in range(6):
            choice = skuscores.index.values.tolist()[i]
            columns = skuscores['columns'][choice]
            name = skuscores['Name'][choice]
            outdf_choice = skuclust[choice]
            outdf_choice = outdf_choice[['cluster', 'of']]
            labels = outdf_choice['cluster']
            outdf_choice.rename(columns = {'cluster': str(choice) + '_cluster', 'of' : str(choice) + '_of'}, inplace = True)

            outdf = outdf.join(outdf_choice)
            #if choice > 6:
                #featuredict = sku.featureweights[choice]
                #featureweightdf = pd.DataFrame.from_dict(featuredict)
                #featureweightdf['id'] = choice
                #featureweightdict = featureweightdict.append(featureweightdf, ignore_index = True)
            #transname = skuscores['Name'].loc[choice]
            sku.write_output(choice, labels, columns, textfile, name, image_loc)
        break

    columns = skuscores['columns'].loc[choice]
    name = skuscores['Name'].loc[choice]
    outdf_choice = skuclust[choice]
    outdf_choice = outdf_choice[['cluster', 'of']]
    labels = outdf_choice['cluster']
    outdf_choice.rename(columns = {'cluster': str(choice) + '_cluster', 'of' : str(choice) + '_of'}, inplace = True)

    #if choice > 6:
        #featuredict = sku.featureweights[choice]
        #featureweightdf = pd.DataFrame.from_dict(featuredict)
        #featureweightdf['id'] = choice
        #featureweightdict = featureweightdict.append(featureweightdf, ignore_index = True)
    #transname = skuscores['Name'].loc[choice]
    sku.write_output(choice, labels, columns, textfile, name)

textfile.close()
skuscores[['Name', 'Perc_noise', 'Numclusters','Outliers' , 'DBCV']].to_csv(image_loc + 'SKUFinalScoring.csv')
outdf.to_csv(image_loc + 'SKUFinalModelOptics.csv')
featureweightdict.to_csv(image_loc  + 'SKUFinalModelFeatureWeights.csv')
