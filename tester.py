from matrix import *

a = Matrix(3, 2)
a.randomize()
a.show()
b = Matrix(3, 2)
b.randomize()
b.show()
c =  a + b
c.show()
d =  a * b;
d.show();
e = a + 5.32
e.show()
f = a * 10
f.show()


b = Matrix(2, 3)
b.randomize();
a.show()
b.show()
h =  a ** b
h.show()
i = Matrix(3, 3)
i.randomize();
i.show()
i = i.transpose();
i.show();