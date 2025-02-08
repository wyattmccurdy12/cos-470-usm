"""
Curse of dimensionality

The curse of dimensionality refers to "various phenomena that arise when analyzing 
and organizing data in high-dimensional spaces." [1]
This has many implications for machine learning, such as the fact that the number of data points required 
to fill the space grows exponentially with the dimension. 
This is why many machine learning algorithms perform poorly in high dimensions.

@Julia: 1.11.3
@OS: Linux (x96_64) (Ubuntu 20.04)
@Package: LinearAlgebra (norm)
@CPU: Intel Core i7 - 12800 MHz
@Memory: 32.6 GiB
@Author: quinlan
"""

using LinearAlgebra: norm

# Set up: d = dimension, n = number of samples
d = 3
n = 100
X = rand(d, n)
Y = rand(d, n)


# Test 1
f(X,Y) = mapreduce(j -> norm(X[:,j] - Y[:,j]), +, 1:size(X,2)) / size(X,2)
f(X,Y)


# Test 2: If you prefer loops
D = 0.0
for j = 1:n
  D += norm(X[:,j] - Y[:,j])
end
D /= size(X,2)

# Test 3
# Size of hypercube
# L = length of side.
# k = number of points in the hypercube
# n = number of points in the dataset
# p = number of dimensions

k = 10
p = [2,4,8,16,32,64,128,256,512,1024]
L = @. (k/n)^(1/p)
# Conclusion: The size of the hypercube grows exponentially with the number of dimensions.
# Taking up the entire [0,1]^p space.



[1] https://en.wikipedia.org/wiki/Curse_of_dimensionality
