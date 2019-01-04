# Clustering
This section will detail a little about the OPTICS algorithm, the outlier detection, and the clustering validation score.

The optics module used in this project was originally taken from the scikit-learn development version. It was modified to include in-line outlier detection and cluster validation scores.

## OPTICS
To understand how it creates this plot you must understand a few definitions.  You must first understand how DBSCAN works, the parameters it uses, and the difference between core and boundary points. I'll leave that out of the scope of this article. We'll also add a couple more definitions <br>

Core Distance- the minimum epsilon to make a distinct point a core point, given a finite MinPts parameter. <br>
Reachability Distance- the reachability-distance of an object p with respect to another object o is the smallest distance such that p is directly density-reachable from o if o is a core object. It also cannot be smaller than the core distance. <br>

![Core / Reachability Distance](/reports/figures/optics/Core_Reach.png)

Although the MinPts parameter is used in these calculations, the idea is that it would not have much of an impact because all distances would scale at the roughly the same rate. 
We will use these definitions to  create our reachability plot, which will then be used to extract the clusters. First, we start out by calculating the core distances on all data points in the set. Then we will loop through the entire data set, and update the reachability distances, processing each point only once. However, when we process a point we set in stone its location on the reachability plot, as well as its reachability distance. For that reason, we can only update reachability distances on points that have not been processed yet. The next data point chosen will be that which has the closest reachability distance. This is how the algorithm keeps clusters in the same location of the reachability plot. An example of the raw reachability plot from this project is shown below. 

![OPTICS](/reports/figures/optics/local_reach.png)


The next step will be to extract the actual cluster labels from the plot. The most common way of doing this is by searching for "valleys" in the plot, using local minimums and maximum. A few more parameters  will come into play here, but they are pretty intuitive. The parameters are pretty much all directly related to how strong, large, or small you would like your clusters to be. In this module, the only parameter changed from the scikit-learn defaults was min_samples which was set to .03, to ensure all clusters were of significant size.

This is an example of the reachability plot generated from this project.

![OPTICS Example](/reports/figures/optics/optics_example.png)


## OPTICS-OF
Another interesting aspect of the OPTICS algorithm is an extension of it used for outlier detection, called OPTICS-OF (OF for Outlier Factor). This will give an outlier score to each point, that is a comparison to its closest neighbors rather than the entire set. This is a unique type of outlier detection because of this "local" principle. We can also use some of the previous calculations while creating it. 

First, what it does is create a new measure ,"local reachability density" , that is the inverse of the average reachability( with respect to the point you are calculating) of the MinPts-neighbors.

![local reachability](/reports/figures/optics/local_reach.png)

Once we have done this for every point, we will calculate the Outlier Factor. To do so, you take the average of the ratios of the MinPts-neighbors to that specific point, shown below.

![outlier factor](/reports/figures/optics/outlier_factor.png)

The "local" component of OPTICS-OF is key to what separates it apart from other outlier detection methods, as it tries to account for the neighborhood of the specific option. It is also able to give it a relative outlier score, as opposed to just a binary value. An example of the local outlier factor, plotted in the same order as reachability, in shown below. All points with a score higher than two are shown in red on the 3D graph.

![Local Outlier Graph](/reports/figures/optics/LocalOutlierGraph.png)

![Local Outlier 3D](/reports/figures/optics/localoutlier3D.png)


## Cluster Validation
The Density Based Clustering Validation (DBCV) score was used to evaluate the effectiveness of the transformation and clustering.The process for calculating the density based clustering validation score is as follows.
<br>
for each cluster: <br>
   -computing the mutual reachability distance between all points using core distance and pairwise distances. The mutual reachability distance is the maximum between the core distances of both points and the distance between points.Many of the pre-computed items we used in clustering can be re-used here  <br>
   -creating a mutual reachability distance graph G <br>
   -creating the minimum spanning tree of G <br>
   -calculating the density sparseness (DSC). This is the maximum distance from the minimum spanning tree within the cluster. <br>
   -calculating the density separation (DSPC). This is the minimum distance between this cluster and other clusters, using the minimum spanning tree.  <br>
   -calculate the validity index <br>

   validity index function:
   ![vailidity function](reports/figures/OPTICS/validity.png)

   DSPC - Density Separation of a pair of clusters
   DSC - Density Sparseness


then we use the scores of all clusters to determine the weighted average of all clusters.
For more details on each of these functions, please read the document [here](/Literature/DBCV.pdf)
