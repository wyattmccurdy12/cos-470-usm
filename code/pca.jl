"""
Principal Component Analysis (PCA)
  This implementation uses MultivariateStats package [1].
  Equivalently, we can use LinearAlgebra.svd.
  
  [1] https://juliastats.org/MultivariateStats.jl/dev/
  
  @Julia: 1.11.3
  @OS: Linux (x96_64) (Ubuntu 20.04)
  @CPU: Intel Core i7 - 4800 MHz
  @Memory: 32.6 GiB
  @Author: james.quinlan
"""

using MultivariateStats

X = [1 2 3 1;
     2 3 1 5;
     3 1 2 6;
     4 5 1 7;
     0 4 0 8]

M = MultivariateStats.fit(PCA, X', pratio=1.0) 
loadings = projection(M)               
scores = MultivariateStats.transform(M, X')      
variances = principalvars(M)
proportion_var = variances / sum(variances)
Xrec = reconstruct(M, scores)
