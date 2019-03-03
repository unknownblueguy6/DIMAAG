from matrix import *
from dimaag import *

# def doubleIt(x):
#     return 2*x

# a = Matrix(3, 2)
# a.randomize()
# a.show()
# a.mapfunc(doubleIt)
# a.show()
# b = Matrix(3, 2)
# b.randomize()
# b.show()
# c =  a + b
# c.show()
# d =  a * b;
# d.show();
# e = a + 5.32
# e.show()
# f = a * 10
# f.show()


# b = Matrix(2, 3)
# b.randomize();
# a.show()
# b.show()
# h =  a ** b
# h.show()
# i = Matrix(3, 3)
# i.randomize();
# i.show()
# i = i.transpose();
# i.show();

nn = NeuralNetwork(2, [2], 1)
s  = nn.feedforward([0, 1])
print(s)