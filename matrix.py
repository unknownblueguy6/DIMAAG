import random

class Matrix:
    def __init__(self, r, c):
        self.rows = r
        self.cols = c
        self.data = [[0 for j in range(self.cols)] for i in range(self.rows)]
        
        # else:
        #     for i in self.rows:
        #         self.data.append([])
        #         for j in self.cols:
        #             self.data[i].append(d[i][j])
    
    def mapfunc(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])
    
    def randomize(self):
        random.seed()
        for i in range(self.rows):
            for j in range(self.cols):
                #element = 2 * random.random() - 1
                self.data[i][j] = random.randint(0, 9)
                
    def copy(self):
        copy = Matrix(self.rows, self.cols)

        for i in range(self.rows):
                for j in range(self.cols):
                    copy.data[i][j] = self.data[i][j]
        return copy

    def show(self):
        for row in self.data:
            s = ''
            for element in row:
                s += str(element) + ' '
            print(s)
        print()
    
    def __add__(self, other):
        copy = self.copy()
        
        if(type(other) == Matrix):
            for i in range(copy.rows):
                for j in range(copy.cols):
                    copy.data[i][j] += other.data[i][j]

        else:
            for i in range(copy.rows):
                for j in range(copy.cols):
                    copy.data[i][j] += other
        
        return copy

    def __mul__(self, other):
        copy = self.copy()
        
        #Hadamard product
        if(type(other) == Matrix):
            for i in range(copy.rows):
                for j in range(copy.cols):
                    copy.data[i][j] *= other.data[i][j]

        else:
            for i in range(copy.rows):
                for j in range(copy.cols):
                    copy.data[i][j] *= other
        
        return copy

    def __pow__(self, other):
        ans = Matrix(self.rows, other.cols)

        #Normal product

        for i in range(ans.rows):
            for j in range(ans.cols):
                sum = 0

                for k in range(self.cols):
                    sum += self.data[i][k] * other.data[k][j]

                ans.data[i][j] = sum

        return ans

    def transpose(self):
        trans = Matrix(self.cols, self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                trans.data[j][i] = self.data[i][j]
        
        return trans