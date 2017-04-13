
local function starcraft(opt)
	opt = opt or {}
	nbInput = opt.nbInput
	nbAgent = opt.nbAgent
	nbActions = opt.nbActions
	neuronesPerLayer = opt.neuronesPerLayer
	nbHiddenLayer = opt.nbHiddenLayer
	pathToOldNetwork = opt.pathToOldNetwork
	networktype = opt.networktype

	local perf = require 'twrl.perf'({nIterations = 1000, windowSize = 10})

	local util = require 'twrl.util'()
	
	local agent = {}
	if networktype == "mlp" then
		agent ={
		   policy = "categorical",
		   learningUpdate = "reinforce",
		   model = "mlp"

		}
	else
		agent = {
		   policy = "egreedy",
		   learningUpdate = "tdLambda",
		   model = "qFunction"
		}
	end



	local actionSpace = {}
	actionSpace.name = "Discrete"
	actionSpace.n = nbAgent*nbActions

	local stateSpace = {}
	stateSpace.name = "Box"
	stateSpace.shape = {}
	stateSpace.shape[1] = nbInput
	stateSpace.high = {}
	stateSpace.low = {}
	for i=1,nbInput do
		stateSpace.high[i] = 999
		stateSpace.low[i] = 0
	end
	local agentOpt = {}
	agentOpt.stateSpace = stateSpace
	agentOpt.actionSpace = actionSpace
	agentOpt.nIterations = opt.nIterations
	agentOpt.model = agent.model
	agentOpt.policy = agent.policy
	agentOpt.learningUpdate = agent.learningUpdate
	agentOpt.envDetails = util.getStateAndActionSpecs(stateSpace, actionSpace)
	agentOpt.nbAgent = nbAgent
	agentOpt.nHiddenLayerSize = neuronesPerLayer
	agentOpt.nbLayers = nbHiddenLayer
	agentOpt.NN = pathToOldNetwork
	
	
	agentOpt.client = opt.client
	agentOpt.instanceID = instanceID
	agentOpt.epsilon = opt.epsilon
	agentOpt.epsilonMinValue = opt.epsilonMinValue
	agentOpt.epsilonDecayRate = opt.epsilonDecayRate
	agentOpt.gamma = opt.gamma

	agentOpt.lambda = opt.lambda
	agentOpt.std = opt.policyStd
      	agentOpt.learningType = opt.learningType
        function agentOpt.randomActionSampler() return math.floor(math.random()*nbAgent*nbActions) end

	agentOpt.tdLearnUpdate = opt.tdLearnUpdate
	agentOpt.baselineType = opt.baselineType
	agentOpt.stepsizeStart = opt.stepsizeStart
	agentOpt.policyStd = opt.policyStd
	agentOpt.beta = opt.beta
	agentOpt.gradClip = opt.gradClip
	agentOpt.weightDecay = opt.weightDecay
	agentOpt.numTiles = opt.numTiles
	agentOpt.numTilings = opt.numTilings
	agentOpt.relativeAlpha = opt.relativeAlpha
	agentOpt.optimAlpha = opt.optimAlpha
	agentOpt.optimType = opt.optimType
	agentOpt.verboseUpdate = opt.verboseUpdate	
	agentOpt.initialWeightVal = opt.initialWeightVal
	
	local agent = require 'twrl.agent.baseAgent'(agentOpt)

	function predict(state)
		return agent.selectAction(state)
	end

	function reward(state, action, nIter, reward, terminal , nextState ,nextAction)
		rewardOpt = {state = state, action = action, reward = reward, nextState = nextState, nextAction = nextAction, terminal = terminal, nIter = nIter}
		perf.addReward(nIter, reward, terminal)
		if action ~= nil then
			agent.reward({state = state, action = action, reward = reward, nextState = nextState, nextAction = nextAction, terminal = terminal, nIter = nIter})
		end
	end
	function save(path)
		agent.save(path)
	end


	return {
		predict = predict,
		reward = reward,
		save = save
	}
end
return starcraft

