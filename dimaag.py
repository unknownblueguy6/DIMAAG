from matrix import *

class NeuralNetwork:
    def __init__(self, noOfInputNeurons, hiddenLayerNeuronsNos, noOfOutputNeurons):
        self.noOfInputNeurons      = noOfInputNeurons
        self.noOfOutputNeurons     = noOfOutputNeurons
        self.noOfHiddenLayers      = len(hiddenLayerNeuronsNos)
        self.hiddenLayerNeuronsNos = []
        
        for hiddenLayer in hiddenLayerNeuronsNos:
            self.hiddenLayerNeuronsNos.append(hiddenLayer)

        self.inputNeuronMatrix   = Matrix(noOfInputNeurons, 1)
        self.outputNeuronMatrix  = Matrix(noOfOutputNeurons, 1)
        self.hiddenLayerMatrices = []
        
        for i in range(self.noOfHiddenLayers):
            self.hiddenLayerMatrices.append(Matrix(self.hiddenLayerNeuronsNos[i], 1))

        self.weightsMatrices = []

        for j in range(self.noOfHiddenLayers + 1):
            if   j == 0:
                self.weightsMatrices.append(Matrix(self.hiddenLayerNeuronsNos[j], self.noOfInputNeurons          ))
            
            elif j == self.noOfHiddenLayers:
                self.weightsMatrices.append(Matrix(self.noOfOutputNeurons       , self.hiddenLayerNeuronsNos[j-1]))
            
            else:
                self.weightsMatrices.append(Matrix(self.hiddenLayerNeuronsNos[j], self.hiddenLayerNeuronsNos[j-1]))



