# Density Based Order Mining Tool (D-BOM)
(pronounced dä bŏm)

The 80/20 rule in warehousing/logistics highlights that 80% of your volume will come from 20% of your SKU's. However, this results in that last 20%  taking up a large majority of the space and labor of the operations. This is true not only because of the lower volume, but the inconsistencies that come along with these SKU's. They may be highly seasonal, or ordered in odd fashions(IE ordered rarely but in large quantities), or ordered with odd items(IE items not typically ordered with the same SKU Type). Understanding how your SKU's act in similar fashion, as well as understanding which items are irregular can help in reducing this effect.

This project is meant to be a knowledge discovery application, that will take an order file from a warehouse/manufacturing facility, and extract meaningful clusters of SKU's from the data for extra analysis. Taking in the order file, it will look for stratifications between SKUs pertaining to: <br>
-Units of handling <br>
-Volume <br>
-SKU Types <br>
-Seasonality <br>
-Order Commonality <br>
-Facility <br>

Each of these analysis are separate transformations, but the tool will look through all combinations of them individually. These will be called 'transformation combos', and will consist of combining different transformations, and then running feature reduction techniques on them. Each transformation combo will be given a final score, and the results from the top scoring combos will be saved and output. For more detail on the specific transformations, please see the feature extraction [REAMDE](/feat_extraction/README.md). Because there is likely to be a lot of noise, trying to put every SKU in a cluster may degrade the meaningfullness of the clusters. For that reason a density based approach was taken, because we would like to identify different groups we can take different actions on, not necessarily focusing on the result on an individual level.

After applying each transformation, they will be applied again in a new fashion. In this iteration, a feature weight learning technique will be applied to the transformed data to try and identify feature importance in clustering. The feature weight learning is based on reducing "Fuzziness" of a similarity matrix. Gradient Descent is used to identify the optimal feature weights, and they will be normalized. These feature weights will then be applied to the scaled data before applying PCA, in an attempt to focus PCA on fitting to the more important variables. For more information on the feature weight learning component, check out my other repo "Fuzzy_Feature_Weighting".

The algorithm employed in this application is an extension of DBSCAN, called [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) (Ordering Points to Identify Cluster Structure). The implementation will be based from the current Scikit-Learn development version, with some added functionality for this project. Along with generating cluster structure, this tool will attempt to identify outliers along the way. The optics algorithm allows for identifying outliers in line with clustering after it computes the core and reachability distances, through the use of [OPTICS-OF](https://pdfs.semanticscholar.org/9d0b/5e35a23117972730fed590ba0a948ad11346.pdf?_ga=2.159976125.1449441537.1543467372-1547328852.1543467372). The OPTICS-OF algorithm works by looking for 'local' outliers rather than outliers of the entire data set.

The final score will be based on the Density Based Cluster Validation Index [DBCV](https://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96). DBCV looks to measure the least dense region inside a cluster and the most dense region between clusters. This is a good technique for non-globular clusters, as we will most likely see in the data from a density based approach. Variables used while calculating the OPTICS plot can also be used in creating the DBCV, so it can also be created with less cost than without the algorith. For more information on the optics algorithm, the local outlier factor, and the DBCV, please see the optics [README](/optics/README.md)

Because so many transformation combinations must be applied, and feature weights must be learned, this can take a few hours. For that reason, I have automated the output to only give results on the top 10 transformation combinations. This can be changed fairly easily within the main source file. It will write the results into a markdown file, and attempt to extract meaningful information about each of the clusters found in each transformation. It will also provide in csv form: <br>
-each sku, along with their statistics used and what cluster they were assigned in each transformations <br>
-the learned feature weights of each transformation
-the total score/stats of each transformation

the output is an entire folder comprised of the excel files, the figures for each transformation, and a markdown file.

Example Results:
The example posted here is from a project I worked on quite a while back from a fruit packaging facility. I cannot provide the raw materials, but can show a example output. In that folder, I have provided a README that also shows a little what I would take away from this run.
See /reports/output
