import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.metrics import pairwise_distances
import math
import matplotlib.pyplot as plt


weightdict = {0: '', 1: ' Weighted'}

def remove_dups(sku, data):
    """remove duplicates and store them separately.

    Parameters
    ----------
    sku : class
        sku class to store data
    data : array
        array to search

    Returns
    -------
    array
        array with suplicates removed

    """
    #place duplicates extremely close, we cannot have duplciates for local outlier factor but would like to maintain density
    df= pd.DataFrame(data)
    dups = df.duplicated()
    #noise = np.random.randn(dups[dups == True].shape[0],df.shape[1] )/120
    #data[dups] = data[dups] + noise
    sku.indices = sku.basedf.index.values[dups == False]
    df.drop_duplicates(inplace = True)

    return df.values

def return_non_binary(x):
    validcols = None
    for i in np.arange(x.shape[1]):
        if np.unique(x[:, i]).ravel().shape[0] > 1:
            if validcols == None:
                validcols = [i]
            else:
                validcols.append(i)
    return validcols



def find_components( pca, tr, ratio = .85):
    """find number of components needed to retain a a certain amount of variance.

    Parameters
    ----------
    pca : object
        pca class used
    tr : object
        transformation function of class
    ratio : float
        amount of ratio required

    Returns
    -------
    int
        number of components

    """
    components = pca.components_
    ex_var = pca.explained_variance_ratio_
    # select components
    sums = ex_var.cumsum()
    n_comps = 0
    for i, s in enumerate(sums):
        if s > ratio:
            n_comps = i
            print('{} components needed to fit {:.2f} of variance'.format(n_comps + 1, s))
            tr.varaince_ = s
            break
    return n_comps + 1



def weighted_euclidean(X, V, weights):
    """Weighted euclidean distance function

    Parameters
    ----------
    X : array
        first object
    V : array
        second object
    weights : array
        feature weights

    Returns
    -------
    float
        weighted distance

    """
    dists = X- V
    return np.sqrt(np.sum((dists * weights) **2))


def single_delta(X, V, F):
    """Distance using one single parameter

    Parameters
    ----------
    X : type
        Description of parameter `X`.
    V : type
        Description of parameter `V`.
    F : type
        Description of parameter `F`.

    Returns
    -------
    type
        Description of returned object.

    """
    d = X[F] - V[F]
    return d


def calc_beta(X, d):
    """calculate beta calue for feature weight learning

    Parameters
    ----------
    X : array
        data set
    d : array
        distance matrix

    Returns
    -------
    float
        beta value

    """
    n = X.shape[0]
    for b in np.linspace(0,1,10000):
        p = 1/(1+b*d)
        p = np.triu(p, 1)
        if (2 / (n*(n-1))) *np.sum(p)< .5:
            return b


def return_weights(X, b, d, mincols):
    """returns learned feature weights, given the data set, beta, the distance matrix and the minimum number of columns

    Parameters
    ----------
    X : array
        data set
    b : float
        beta value
    d : array
        distance atrix
    mincols : int
        minimum number of columns to return that have weights

    Returns
    -------
    array
        learned feature weights

    """
    max_iter = 100
    threshold= .00001
    w= np.empty((1,X.shape[1]))
    w.fill(1)
    p_1 = 1/(1+b*d)
    n = X.shape[0]
    E_old = 1
    for i in np.arange(0, max_iter):
        d = pairwise_distances(X,X, metric = weighted_euclidean, **{'weights':w})
        grad_w = np.empty((1,X.shape[1]))
        part_pq = -b/((1+b*d)**2)
        p = 1/(1+b*d)
        E = (2/(n*(n-1))) * np.sum(np.triu(.5*((p*(1-p_1) + p_1*(1-p))), 1))
        if E_old - E < threshold:
            break
        E_old = E
        part_eq = (1-2*p_1)
        w_valid = np.where(w > 0)[1]

        if w_valid.shape[0] == mincols:
            break

        for j in w_valid:
            d_w = pairwise_distances(X, X, metric = single_delta, **{'F':j})
            part_w = w[0, j]*(d_w)**2 / d
            part_w = np.triu(part_w, 1)
            grad_w_j = 1/(n*(n-1)) * part_eq * part_pq * part_w
            grad_w_j = np.triu(grad_w_j, 1)
            grad_w[ 0, j] = np.nansum(grad_w_j)
        grad_w = grad_w * 50
        w = w-grad_w
        w = w.clip(min=0)
        #if i %10 == 0 and i > 0:
            #print("Iteration {} Finished".format(i))
            #print("Weights : {} ".format(w))
            #print("Function Improvement : {}".format(E_old - E))

    wmax = np.max(w)
    w = w / wmax
    print("{} Iterations Required".format(i))
    return w



def return_weighted(sku, X,columns, mincols = 3, sample_size = .15, influence = 4):
    """takes in data set and completes entire process of feature weight learning.
    1. Calculate Pairwise Distances
    2. Calculate Beta
    3. Learn Feature weights through gradient descent

    Parameters
    ----------
    sku : class transformation
        class transformation where to add feature weights
    X : array
        data set
    columns : list
        column names
    mincols : int
        minimum number of columns to returdn
    sample_size :  float
        fraction of dataset to use in feature weight learning
    influence : float
        amount of influence the feature weights will return, higher = less influence

    Returns
    -------
    array
        weighted dataset

    """
    print("Weighting Columns {} ".format(columns))
    numsample = math.ceil(sample_size * X.shape[0])
    sample = np.random.choice(X.shape[0],numsample, replace = False )
    X_S = X[sample]
    d = pairwise_distances(X_S, X_S, metric = 'euclidean')
    print("calculating Beta Value")
    b = calc_beta(X_S, d)
    print("calculating feature weights")
    w = return_weights(X_S, b, d, mincols)
    w = w.reshape(-1,1)

    featuredict = {}
    for i in np.arange(0, X.shape[1]):
        featuredict[columns[i]] = np.asscalar(w[i])
        X[:, i] = X[:, i] *   w[i]
    #print("Feature Weights: {}".format(featuredict))
    sku.featureweights[sku.transformation] = featuredict

    return X




pca = PCA(n_components = None)

class DATA():
    """class to clean and return a usable data set

    Parameters
    ----------
    data : dataframe
        raw data read into pandas dataframe

    Attributes
    ----------
    data

    """
    def __init__(self, data):
        self.data = data

    def clean(self):
        """replace column names, convert shipdate to datetime, filter on lines wwe care about.

        Parameters
        ----------


        Returns
        -------
        dataframe
            cleaned dataframe

        """
        data = self.data
        data.columns = [x.replace( ' ', '') for x in data.columns]
        data['ShipDate'] = pd.to_datetime(data['ShipDate'])
        data = data[data.LooseUOH1Lines + data.FullUOH2Lines + data.FullUOH3Lines > 0]

        return data


class SKU():
    """Short summary.

    Parameters
    ----------
    data : dataframe
        initial data to create full set

    Attributes
    ----------
    base : dataframe
        raw data set
    data_ : dataframe
        base dataframe that includes unit of handling
    skutype1 : dataframe
        one hot encoded sku types
    skutype2 : dataframe
        one hot encoded sku type 2
    skutype3 : dataframe
        one hot encoded sku type 3
    seasonality_ : dataframe
        ratio of lines by quarter, per sku
    commonality_ : dataframe
        ratio of lines per sku type 1 in same order
    savedf : dataframe
        final dataframe with all features
    featureweights : dict
        learned deature weights for every transformation

    """

    def __init__(self, data):
        """set up all required aspects that transformations will use.

        Parameters
        ----------
        data : dataframe
            cleaned dataset

        Returns attributes of SKU class that will be used.
        -------

        """
        self.base = data
        numlines = data.groupby(by = 'SKU')['Lines'].sum()
        totaldf = numlines
        collist = ['Lines']
        cols = ['LooseUOH1Lines', 'FullUOH2Lines', 'FullUOH3Lines']
        col2 = ['LooseUOH1s', 'FullUOH2s', 'FullUOH3s']

        #Avg share of lines by UOH
        for i, col in enumerate(cols):
            newdf = data.groupby(by = 'SKU')[col].sum() / numlines
            totaldf = pd.concat([totaldf, newdf], axis = 1)
            collist.append(col)
            newdf_2 = data.groupby(by = 'SKU')[col2[i]].sum() / data.groupby(by = 'SKU')[col].sum() # Units per line
            totaldf = pd.concat([totaldf, newdf_2], axis = 1)
            colname = str(col2[i]) + 'per' + str(col)
            collist.append(colname)



        #Unique shipdays
        cols = ['ShipDate']
        for i in cols:
            newdf = data.groupby(by = 'SKU')[i].nunique()
            totaldf = pd.concat([totaldf, newdf], axis = 1)
            collist.append(i)

        totaldf.columns = collist

        totaldf.rename(columns = {'LooseUOH1sperLooseUOH1Lines': 'UOH1perLine','FullUOH2sperFullUOH2Lines': 'UOH2perLine', 'FullUOH3sperFullUOH3Lines': 'UOH3perLine' }, inplace = True)

        #Percent of days active
        #mindate =  data.groupby(by = 'SKU')['ShipDate'].min()
        #mindate.rename('mindate', inplace = True)
        #maxdate =  data.groupby(by = 'SKU')['ShipDate'].max()
        #days_active = pd.concat([mindate, maxdate], axis = 1)
        #days_active['numdays'] = (pd.to_datetime(days_active['ShipDate']) - pd.to_datetime(days_active['mindate']))
        #days_active['numdays'] = days_active['numdays'].apply(lambda x: x.days + 1)
        #totaldf['numdays'] = days_active['numdays']
        #totaldf['perc_active'] = totaldf['ShipDate'] / totaldf['numdays']
        #totaldf['linesperday'] = totaldf['Lines'] / totaldf['numdays']
        #totaldf.fillna(0, inplace = True)
        #totaldf.drop(['numdays', 'perc_active'], axis = 1, inplace = True)

        #save the SKU features so we can look at those details as well when comparing
        labeldf = data.groupby(by = ['SKU', 'SKUType1']).count().reset_index()
        totaldf.fillna(0, inplace = True)
        #self.data_ = totaldf
        self.trandict = {}
        self.trandict['UOH'] = totaldf.drop(columns = ['Lines', 'ShipDate'])
        #save skutypes one hot encoded
        skutype1_df = pd.get_dummies(labeldf['SKUType1']).set_index(labeldf['SKU'])
        #for col in self.skutype1.columns:
            #self.skutype1[col] = 0
        self.trandict['Type 1'] = skutype1_df

        labeldf = data.groupby(by = ['SKU', 'SKUType2']).count().reset_index()
        self.skutype2 = pd.get_dummies(labeldf['SKUType2']).set_index(labeldf['SKU'])

        labeldf = data.groupby(by = ['SKU', 'SKUType3']).count().reset_index()
        self.skutype3 = pd.get_dummies(labeldf['SKUType3']).set_index(labeldf['SKU'])

        #Save Seasonality
        data['month'] = data['ShipDate'].apply(lambda x: math.ceil(x.month / 3))
        by_month = data.pivot_table(index = 'SKU', columns = 'month', values = 'Lines', aggfunc = np.sum)
        by_month = by_month.divide(numlines, axis = 0)
        by_month.fillna(0, inplace = True)
        self.trandict['Seasonality'] = by_month


        #Save commonality
        collist = ['SKU'] + data['SKUType1'].unique().tolist()
        ocadf = pd.DataFrame(columns = collist)
        for i in data['SKU'].unique():
            orders = data[data.SKU == i]['CustomerOrderID'].unique()
            oca = data[data.CustomerOrderID.isin(orders)].pivot_table(index = 'CustomerOrderID', columns = 'SKUType1', values = 'Lines', aggfunc = np.sum)
            oca = oca.sum(axis = 0)
            oca = oca / oca.sum()
            oca['SKU'] = i
            ocadf = ocadf.append(oca, ignore_index = True)
        collist = ['SKU'] + [x + "_oca" for x in  data['SKUType1'].unique().tolist()]
        ocadf.columns = collist
        ocadf.set_index('SKU', inplace = True)
        ocadf.fillna(0, inplace = True)
        self.trandict['OCA'] = ocadf


        #by facility
        by_facility = data.pivot_table(index = 'SKU', columns = 'FacilityID', values = 'Lines', aggfunc = np.sum)
        by_factility = by_facility.divide(numlines, axis = 0)
        by_facility.fillna(0, inplace = True)
        self.trandict['Facility'] = by_facility





        self.savedf = totaldf.join(skutype1_df).join(by_month).join(ocadf).join(by_facility)
        self.basedf = np.log(totaldf['Lines'])

        self.featureweights = {}


    def scale_base(self, weights, trans):
        """base transformation

        Parameters
        ----------
        weights : binary
            whether to include feature weighting or not

        Returns
        -------
        array
            data to be clustered

        """
        data = self.basedf.copy()
        self.transformation = ''
        for i in trans:
            self.transformation += i + ' '
            data = pd.concat((data,self.trandict[i]), axis = 1)
        print("Applying {} Transformation".format(self.transformation))
        cols = data.columns
        self.columns = cols
        x = StandardScaler().fit_transform(data)
        if weights ==1:
            validcols = return_non_binary(x)
            colnames = [cols[x] for x in validcols]
            w_x = return_weighted(self, x[:, validcols], colnames, 3)
            x[:, validcols] = w_x
        X=pca.fit_transform(x)
        num_comp = find_components(pca, self)
        num_comp = max(3, num_comp)
        #X = remove_dups(X)
        return X[:, :num_comp]


    def custom(self, weights):
        """custom transformation

        Parameters
        ----------
        weights : binary
            whether to include feature weighting or not

        Returns
        -------
        array
            data to be clustered

        """
        data = self.savedf.copy()
        self.transformation = 'Total Features' + str(weightdict[weights])
        print("Applying {} Transformation Combo".format(self.transformation))
        data['Lines'] = np.log(data['Lines'])
        cols = data.columns
        self.columns = cols
        x = StandardScaler().fit_transform(data)
        if weights ==1:
            validcols = return_non_binary(x)
            colnames = [cols[x] for x in validcols]
            w_x = return_weighted(self, x[:, validcols], colnames, 3)
            x[:, validcols] = w_x
        X = pca.fit_transform(x)
        numcomp = find_components(pca, self)
        return X[:, :numcomp]


    def write_output(self, ID, labels, columns, textfile, name, image_loc):
        print("'''Summarizing {} Transformation Results'''".format(name))
        textfile.write("## {} Transformation Results \n".format(name))
        #trancoldict = self.featureweights[ID]
        #transcolumns = trancoldict.keys()
        Y = self.savedf
        Y['clust'] = labels
        Y.dropna(inplace = True)



        df = Y.groupby(by = 'clust')[columns].mean()
        groupcols = self.trandict['Type 1'].columns
        df2 = Y.groupby(by = 'clust')[groupcols].sum()
        groupsum = df2[groupcols].sum(axis = 1)

        headers = df.columns
        for col in [x for x in df.columns if x not in groupcols]:
            df[col] = df[col] / Y[col].mean() - 1
        for skutype in groupcols:
            df[skutype] = df2[skutype] / groupsum


        headers = [ x for x in df.columns if x not in groupcols]
        ylab = df.index.values
        fig, (ax1, ax2)= plt.subplots(1, 2,  figsize = (30, 10))
        #fig.subplots_adjust(bottom=0.25,left=0.25) # make room for labels
        heatmap = ax2.pcolor(df[headers], cmap = 'seismic', vmin = -1, vmax = 1)
        cbar = plt.colorbar(heatmap)

        # Set ticks in center of cells
        ax2.set_xticks(np.arange(len(headers)) + 0.5, minor=False)
        ax2.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)

        # Rotate the xlabels. Set both x and y labels to headers[1:]
        ax2.tick_params('both', labelsize = 20)
        ax2.set_xticklabels(headers,rotation=75)
        ax2.set_yticklabels(ylab)
        ax2.set_title('Cluster Feature Heatmap', fontsize = 20)
        ax2.set_ylabel('Clusters', fontsize = 30)

        df2.loc[-1:].plot(kind = 'bar', stacked = True, ax = ax1)
        ax1.set_xlabel('Clusters', fontsize = 30)
        ax1.set_ylabel('Count SKU', fontsize = 30)
        ax1.tick_params('both', labelsize = 20)
        plt.tight_layout()
        fig.savefig(image_loc + ID + 'heatmap.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        textfile.write('![{} counts]({})'.format(ID,  ID + 'heatmap.png'))

        for idx in [x for x in df.index.values if x not in [-1, -2]]:
            textfile.write('\n')
            for col in headers:
                if df[col].loc[idx] > .5:
                    val = df[col].loc[idx] * 100
                    print('Cluster {} has {:02d}% more {} than average.'.format(idx, int(val), col ))
                    textfile.write('Cluster {} has {:02d}% more {} than average. <br> \n'.format(idx, int(val), col ))

                if df[col].loc[idx] < -.5:
                    val = np.abs(df[col].loc[idx]) * 100
                    print('Cluster {} has {:02d}% less {} than average.'.format(idx, int(val), col ))
                    textfile.write('Cluster {} has {:02d}% less {} than average. <br> \n'.format(idx, int(val), col ))
