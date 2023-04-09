Description of files:

SCCS.m - Single Criteria approach based on Cosine Similarity 

SCPC.m - Single Criteria approach based on Pearson Correlation 

SCJC.m - Single Criteria approach based on extended Jaccard Coefficient

SCMMD.m - Single Criteria approach based on Modified Mahalanobis Distance

SCOS.m - Single Criteria approach based on OS measure

SCCRS.m - Single Criteria approach based on Common Rating weight Similarity

Clusters_SI.xlsx - Clusters obtained through clustering users using side information

Clust_SI.m - Clustering with Side Information approach

Clusters_FSI.xlsx - Clusters obtained through clustering users using fuzzy side information

Clust_FSI.m - Clustering with Fuzzy Side Information approach

rating_count.m - calculation of rating count using item criteria obtained through data_preprocessing.m. To be executed before executing NRC-CRS.m and GA-NRC-CRS.m

NRC_CRS.m - Weighted normalized rating count and common rating weight similarity

GA_NRC_CRS.m - GA based RS through weighted normalized rating count and common rating weight similarity
actuserfit.m - function call in GA_NRC_CRS.m for evaluating fitness function. The initialization of GA is done using random binary numbers.   
cross_mut.m - function call in GA_NRC_CRS.m for crossover-mutation operators

The Yahoo! Movies dataset can be obtained from [1]. Original dataset consists of 62156 ratings provided by 6078 users to 976 movies. Dataset description - 

data_movies.txt - 
1st column: user_id
2nd column: evaluation on criterion1
3rd column: evaluation on criterion2
4th column: evaluation on criterion3
5th column: evaluation on criterion4
6th column: overall evaluation
7th column: movie_id
8th column: increasing number of movies

data_preprocessing.m - set of codes for extracting overall and multi-criteria ratings from original data_movies dataset

[1] K.Lakiotaki, N. F. Matsatsinis, and A.Tsoukiàs, “Multi-Criteria User Modeling in Recommender Systems”, IEEE Intelligent Systems, 26 (2), 64 - 76, (2011)
