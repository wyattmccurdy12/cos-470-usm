"""
kNN algorithm for classification and regression.
  
  @Julia: 1.11.3
  @OS: Linux (x96_64) (Ubuntu 20.04)
  @Package: LinearAlgebra (norm)
  @CPU: Intel Core i7 - 4800 MHz
  @Memory: 32.6 GiB
  @Author: james.quinlan
"""

using RDatasets
using Distances
using MLJBase
using StatsBase
using Random

function kNN(X,x,y,k,d = Euclidean())
  n = size(X,2)    
  distances = map(i -> d(x,X[:,i]), 1:n)
  indices = partialsortperm(distances,1:k)
  if typeof(y) == Vector{Union{Float32, Float64}}
      yhat = mean(y[indices])
  else
      yhat = mode(y[indices])
  end
  return yhat
end


# Load data
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])'
y = @. ifelse(iris.Species == "virginica", 1, -1)
c = unique(y)

# Train-test split
Random.seed!(123)
train, test = partition(1:size(X,2), 0.7, shuffle=true)
Xtrain = X[:, train];
Xtest  = X[:, test];
ytrain = y[train];
ytest  = y[test];

# Predictions
yhat = map(i -> kNN(Xtrain, Xtest[:,i], ytrain, 1), 1:size(Xtest,2))
c[argmax(map(i -> sum(yhat .== c[i]),1:lastindex(c)))]

# Metrics
accuracy = sum(yhat .== ytest) / length(ytest)
precision = sum((yhat .== 1) .& (ytest .== 1)) / sum(yhat .== 1)
recall = sum((yhat .== 1) .& (ytest .== 1)) / sum(ytest .== 1)
specificity = sum((yhat .== -1) .& (ytest .== -1)) / sum(ytest .== -1)
f1 = 2 * precision * recall / (precision + recall)
