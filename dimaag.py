from matrix import *

class NeuralNetwork:
    def __init__(self, noOfInputNeurons, hiddenLayerNeuronsNos, noOfOutputNeurons):
        self.noOfInputNeurons      = noOfInputNeurons
        self.noOfOutputNeurons     = noOfOutputNeurons
        self.noOfHiddenLayers      = len(hiddenLayerNeuronsNos)
        self.hiddenLayerNeuronsNos = []
        
        for hiddenLayerNo in hiddenLayerNeuronsNos:
            self.hiddenLayerNeuronsNos.append(hiddenLayerNo)

        self.inputNeuronMatrix   = Matrix(noOfInputNeurons, 1, False)
        self.outputNeuronMatrix  = Matrix(noOfOutputNeurons, 1, False)
        self.hiddenLayerMatrices = []
        
        self.hiddenBiasesMatrices = []
        self.outputBiasesMatrix = Matrix(noOfOutputNeurons, 1)

        for i in range(self.noOfHiddenLayers):
            self.hiddenLayerMatrices.append(Matrix(self.hiddenLayerNeuronsNos[i], 1, False))
            self.hiddenBiasesMatrices.append(Matrix(self.hiddenLayerNeuronsNos[i], 1))

        self.weightsMatrices = []

        for j in range(self.noOfHiddenLayers + 1):
            if   j == 0:
                self.weightsMatrices.append(Matrix(self.hiddenLayerNeuronsNos[j], self.noOfInputNeurons          ))
            
            elif j == self.noOfHiddenLayers:
                self.weightsMatrices.append(Matrix(self.noOfOutputNeurons       , self.hiddenLayerNeuronsNos[j-1]))
            
            else:
                self.weightsMatrices.append(Matrix(self.hiddenLayerNeuronsNos[j], self.hiddenLayerNeuronsNos[j-1]))

    def feedforward(self, inputList):
        self.inputNeuronMatrix = Matrix.toMatrix(inputList)

        for i in range(self.noOfHiddenLayers + 1):

            if   i == 0:
                self.hiddenLayerMatrices[i] = (self.weightsMatrices[i] ** self.inputNeuronMatrix       ) + self.hiddenBiasesMatrices[i] 
            
            elif i == self.noOfHiddenLayers:
                self.outputNeuronMatrix     = (self.weightsMatrices[i] ** self.hiddenLayerMatrices[i-1]) + self.outputBiasesMatrix
            
            else:
                self.hiddenLayerMatrices[i] = (self.weightsMatrices[i] ** self.hiddenLayerMatrices[i-1]) + self.hiddenBiasesMatrices[i]
        
        return (Matrix.toList(self.outputNeuronMatrix))




        



