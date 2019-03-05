from matrix import *
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidInverse(x):
    return math.log(x/(1-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, noOfInputNeurons, hiddenLayerNeuronsNos, noOfOutputNeurons, 
                 activationFunc = sigmoid, activationFuncInverse = sigmoidInverse, activationFuncDerivative = sigmoidDerivative,
                 learningRate = 0.1):
        self.noOfInputNeurons      = noOfInputNeurons
        self.noOfOutputNeurons     = noOfOutputNeurons
        self.noOfHiddenLayers      = len(hiddenLayerNeuronsNos)
        self.hiddenLayerNeuronsNos = []

        self.learningRate = learningRate
        
        self.activationFunc = activationFunc
        self.activationFuncInverse = activationFuncInverse
        self.activationFuncDerivative = activationFuncDerivative
        
        for hiddenLayerNo in hiddenLayerNeuronsNos:
            self.hiddenLayerNeuronsNos.append(hiddenLayerNo)

        self.inputNeuronsMatrix   = Matrix(noOfInputNeurons, 1, False)
        self.outputNeuronsMatrix  = Matrix(noOfOutputNeurons, 1, False)
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

    def feedforward(self, inputsList):
        self.inputNeuronsMatrix = Matrix.toMatrix(inputsList)

        for i in range(self.noOfHiddenLayers + 1):

            if   i == 0:
                self.hiddenLayerMatrices[i] = (self.weightsMatrices[i] ** self.inputNeuronsMatrix       ) + self.hiddenBiasesMatrices[i]
                self.hiddenLayerMatrices[i].mapfunc(self.activationFunc) 
            
            elif i == self.noOfHiddenLayers:
                self.outputNeuronsMatrix     = (self.weightsMatrices[i] ** self.hiddenLayerMatrices[i-1]) + self.outputBiasesMatrix
                self.outputNeuronsMatrix.mapfunc(self.activationFunc)
            
            else:
                self.hiddenLayerMatrices[i] = (self.weightsMatrices[i] ** self.hiddenLayerMatrices[i-1]) + self.hiddenBiasesMatrices[i]
                self.hiddenLayerMatrices[i].mapfunc(self.activationFunc)
        
        return (Matrix.toList(self.outputNeuronsMatrix))

    def backpropogate(self, inputsLists, targetsLists):
        outputErrorMatrix = Matrix(self.noOfOutputNeurons, 1, False)
        
        for inputsList, targetsList in zip(inputsLists, targetsLists):
            outputErrorMatrix += Matrix.toMatrix(targetsList) - Matrix.toMatrix(self.feedforward(inputsList))

        errorMatrices = [outputErrorMatrix]

        #find error for each weights matrix
        for i in range(self.noOfHiddenLayers):
            weightsTranspose = self.weightsMatrices[self.noOfHiddenLayers - i].transpose()
            errorMatrices.append(weightsTranspose ** errorMatrices[i])
        
        for j in range(self.noOfHiddenLayers + 1):
            if j == 0:
                #find derivative of output layer
                self.outputNeuronsMatrix.mapfunc(self.activationFuncInverse)
                self.outputNeuronsMatrix.mapfunc(self.activationFuncDerivative)
                
                hiddenLayerMatrixTranspose = self.hiddenLayerMatrices[self.noOfHiddenLayers - 1 - j].transpose()
                
                delWeightsMatrix = (errorMatrices[j] * self.outputNeuronsMatrix * self.learningRate) ** hiddenLayerMatrixTranspose 
                delBiasesMatrix  = (errorMatrices[j] * self.outputNeuronsMatrix * self.learningRate)

                self.weightsMatrices[self.noOfHiddenLayers - j] += delWeightsMatrix
                self.outputBiasesMatrix += delBiasesMatrix

            elif j == self.noOfHiddenLayers:
                #find derivative of  layer
                self.hiddenLayerMatrices[0].mapfunc(self.activationFuncInverse)
                self.hiddenLayerMatrices[0].mapfunc(self.activationFuncDerivative)
                
                inputNeuronsMatrixTranspose = self.inputNeuronsMatrix.transpose()
                
                delWeightsMatrix = (errorMatrices[j] * self.hiddenLayerMatrices[0] * self.learningRate) ** inputNeuronsMatrixTranspose
                delBiasesMatrix  = (errorMatrices[j] * self.hiddenLayerMatrices[0] * self.learningRate)

                self.weightsMatrices[self.noOfHiddenLayers - j] += delWeightsMatrix
                self.hiddenBiasesMatrices[0]                    += delBiasesMatrix

            else:
                #find derivative of  layer
                self.hiddenLayerMatrices[self.noOfHiddenLayers - j].mapfunc(self.activationFuncInverse)
                self.hiddenLayerMatrices[self.noOfHiddenLayers - j].mapfunc(self.activationFuncDerivative)
                
                inputNeuronsMatrixTranspose = self.inputNeuronsMatrix.transpose()
                
                hiddenLayerMatrixTranspose = self.hiddenLayerMatrices[self.noOfHiddenLayers - 1 - j].transpose()
                
                delWeightsMatrix = (errorMatrices[j] * self.hiddenLayerMatrices[self.noOfHiddenLayers - j] * self.learningRate) ** hiddenLayerMatrixTranspose
                delBiasesMatrix  = (errorMatrices[j] * self.hiddenLayerMatrices[self.noOfHiddenLayers - j] * self.learningRate)

                self.weightsMatrices[self.noOfHiddenLayers - j] += delWeightsMatrix
                self.hiddenBiasesMatrices[self.noOfHiddenLayers - 1 - j] += delBiasesMatrix