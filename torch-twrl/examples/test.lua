local sc = require 'twrl.starcraft'

opt = {}
opt.nbInput = 4   --table size
opt.nbAgent = 5   -- nb unit to control
opt.nbActions = 6 -- nb action per unit
opt.neuronesPerLayer= 200
opt.nbHiddenLayer = 5
opt.networktype = "mlp"
opt.nIterations = 200
opt.epsilon = 0.1 -- RAndom action chance 
opt.epsilonMinValue = 0.0001
opt.epsilonDecayRate = 0.999
opt.gamma = 1
opt.lambda = 0.9
opt.policyStd = 0.1
opt.learningType = "batch"
opt.tdLearnUpdate = "qLearning"
opt.baselineType = "padTimeDepAvReturn"
opt.stepsizeStart = 0.3
opt.policyStd = 0.1
opt.beta = 0.01
opt.gradClip = 5
opt.weightDecay = 0
opt.numTilings = 4
opt.numTiles = 32
opt.relativeAlpha = 0
opt.optimAlpha = 0.9
opt.optimType = "rmsprop"
opt.verboseUpdate= false
opt.initialWeightVal = -0.01
a = sc(opt)
print("*****")
print(sc)
print(a)

possibleState = {}
for i=1,4 do
	possibleState[i] = i
end
print(possibleState)
print(a.predict(possibleState))
reward(possibleState,a.predict(possibleState), 0, 0, false , possibleState ,2 )
print(a.predict(possibleState))
reward(possibleState,a.predict(possibleState), 0, 0, false , possibleState ,2 )
print(a.predict(possibleState))
reward(possibleState,a.predict(possibleState), 0, 0, false , possibleState ,2 )
opt.pathToOldNetwork = "WAS"
b = sc(opt)
