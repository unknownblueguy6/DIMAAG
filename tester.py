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
# f = 10.111 * a
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



nn = NeuralNetwork(2, [4, 4], 1)
print(nn.feedforward([0, 1]))
print(nn.feedforward([1, 0]))
print(nn.feedforward([0, 0]))
print(nn.feedforward([1, 1]))

#XOR Gate
inputs = [
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[0, 0], [0]],
    [[1, 1], [0]]
]

for i in range(10000):
    s = random.randint(0, 3)
    i, t = inputs[s][0], inputs[s][1]
    nn.backpropogate([i], [t])

print(nn.feedforward([0, 1]))
print(nn.feedforward([1, 0]))
print(nn.feedforward([0, 0]))
print(nn.feedforward([1, 1]))