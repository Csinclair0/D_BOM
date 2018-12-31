# Feature Extraction
This document will describe the feature transformation techniques used to try and stratify the sku base into meaningful clusters. It will also describe the general process of feature weight learning. There are six transformations, which will then be repeated after introducing feature weight learning. In this section, UOH refers to "unit of handling", and goes from smallest to largest unit. An example would be UOH1 = Cases, UOH2 = Layers, UOH3 = Pallets. These have to be already entered into the data. The tool is not tasked with creating the initial dataset. Th dataset must have the following fields at minimum for each record.


ShipDate-----object <br>
CustomerOrderID-----int64 <br>
SKU-----object <br>
SkuType1-----object <br>
SKUType2-----object <br>
SKUType3-----object <br>
Lines-----int64 <br>
UOH1s-----float64 <br>
EqUOH2s-----float64 <br>
EqUOH3s-----float64 <br>
LooseUOH1s-----float64 <br>
FullUOH2s-----float64 <br>
FullUOH3s-----float64 <br>
LooseUOH1Lines-----int64 <br>
FullUOH2Lines-----int64 <br>
FullUOH3Lines-----int64 <br>
FullUOH4Lines-----int64 <br>
FullUOH5Lines-----int64 <br>



The six transformations are as follows:

## Unit of handling
Number of lines (Log)
% of UOH mix (ie % UOH1 vs UOH2)
This is then all standard scaled.

## SKU type
 Sku Type1 one hot encoded

## Seasonality
% of lines ordered by quarter

## Commonality
% of SKU types commonly ordered in same order ID.

## Facility
% of lines from each Facility ID

## Custom
Has the added component of each transformation, while using feature weight learning, it is meant to choose the most useful features to include.

# Feature Weight learning
This is based off the work from this research [paper](https://ieeexplore.ieee.org/abstract/document/993562). The attempt is to identify which features are important at making your data points better for clustering. The implementation here is tailored to this solution, but I have built a stand alone library as well titled "feature_learning", where you can find more information about it. In this application I only weight non-binary features(IE Skutypes) because the algorithm will natural gravitate towards choosing those. I also don't completely eliminate the variables it reduces, but instead reduce the weights by a given factor each time.
