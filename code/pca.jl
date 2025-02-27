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
