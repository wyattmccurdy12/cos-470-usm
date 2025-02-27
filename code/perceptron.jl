"""
The Perceptron
  
  @Julia: 1.11.3
  @OS: Linux (x96_64) (Ubuntu 20.04)
  @CPU: Intel Core i7 - 4800 MHz
  @Memory: 32.6 GiB
  @Author: james.quinlan
"""

function perceptron(X,y)
    n = lastindex(y)
    w = rand(size(X,2)+1)
    m = 1
    while m > 0
        m = 0  
        for i = 1:n 
            x = X[i, :]
            if w'*[x;1] * y[i] <= 0
                w += y[i] * [x;1]
                m += 1
            end
        end
    end
    return w
end

# Test Case
X = [
    2.0  2.0;  # Class 1
    3.0  2.2;  # Class 1
    2.5  3.0;  # Class 1
    3.2  2.8;  # Class -1 
    3.8  2.5;  # Class -1
    2.6  3.5   # Class -1
]
y = [1, 1, 1, -1, -1, -1]
perceptron(X,y)
