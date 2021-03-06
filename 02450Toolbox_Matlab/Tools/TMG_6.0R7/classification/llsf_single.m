function labels_as=llsf_single(A, Q, clusters, labels, l, method, svd_method, clsi_method)
% LLSF_SINGLE - Linear Least Squares Fit for single-label 
% collections [2]
%   LABELS_AS=LLSF_SINGLE(A, Q, CLUSTERS, LABELS, L, METHOD, 
%   SVD_METHOD, CLSI_METHOD) classifies the columns of Q 
%   with the Linear Least Squares Fit classifier [2] using 
%   the pre-classified columns of matrix A with labels LABELS 
%   (cell array of vectors of integers). CLUSTERS is a 
%   structure defining the classes. METHOD is the method
%   used for the approximation of the rank-l truncated SVD, 
%   with possible values:
%       - 'clsi': Clustered Latent Semantic Indexing [3].
%       - 'cm': Centroids Method [1].
%       - 'svd': Singular Value Decomosition.
%   SVD_METHOD defines the method used for the computation 
%   of the SVD, while CLSI_METHOD defines the method used for 
%   the determination of the number of factors from each class 
%   used in Clustered Latent Semantic Indexing in case METHOD 
%   equals 'clsi'. 
%
%   REFERENCES:
%   [1] H. Park, M. Jeon, and J. Rosen. Lower Dimensional 
%   Representation of Text Data Based on Centroids and Least 
%   Squares. BIT Numerical Mathematics, 43(2):427?448, 2003.
%   [2] Y. Yang and C. Chute. A Linear Least Squares Fit 
%   Mapping Method for Information Retrieval from Natural 
%   Language Texts. In Proc. 14th Conference on Computational 
%   Linguistics, pages 447?453, Morristown, NJ, USA, 1992. 
%   [3] D. Zeimpekis and E. Gallopoulos, "Non-Linear Dimensional 
%   Reduction via Class Representatives for Text Classification".  
%   In Proc. 2006 IEEE International Conference on Data Mining 
%   (ICDM'06), Hong Kong, Dec. 2006.  
%
% Copyright 2011 Dimitrios Zeimpekis, Eugenia Maria Kontopoulou, Efstratios Gallopoulos

error(nargchk(8, 8, nargin));
labels_as=llsf_single_p(A, Q, clusters, labels, l, method, svd_method, clsi_method);