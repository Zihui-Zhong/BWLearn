local sc = require 'twrl.starcraft'

opt = {}
opt.nbInput = 10   --table size
opt.nbAgent = 1  -- nb unit to control
opt.nbActions = 10 -- nb action per unit
opt.neuronesPerLayer= 200
opt.nbHiddenLayer = 5
opt.networktype = "ass"
opt.nIterations = 200
opt.epsilon = 0.1 -- RAndom action chance 
opt.epsilonMinValue = 0.0001
opt.epsilonDecayRate = 0.999
opt.gamma = 1
opt.lambda = 0.9
opt.policyStd = 0.1
opt.learningType = "noBatch"
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
opt.tdLearnUpdate = "qLearning"
local a = sc(opt)
print("*****")
print(sc)
print(a)

possibleState = {}
for i=1,10 do
	possibleState[i] = i
end
print(possibleState)


local action =a.predict(possibleState)
local newAction =a.predict(possibleState)
local oldRandom = math.floor(math.random()*10)
local nIter = 1
for i=1,10000 do 
	newAction = a.predict(possibleState)

	local reward = -1
	if newAction == 1 then
		reward = 1
	else
		reward = 0
	end
	print(newAction,reward)
	if i %100 == 0 then 
		a.reward(possibleState,newAction, nIter,reward , true , possibleState ,action )
		nIter = nIter+1
	else
		a.reward(possibleState,newAction, nIter,reward , false , possibleState ,action )
	end
	action = newAction
 	oldRandom = thisRandom
end


opt.pathToOldNetwork = "WAS"
b = sc(opt)

