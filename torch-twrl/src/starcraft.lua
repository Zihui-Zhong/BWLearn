
local function starcraft(x,y,nbOut)
	local perf = require 'twrl.perf'({nIterations = 1000, windowSize = 10})

	local util = require 'twrl.util'()
	local agent = {
	   policy = "categorical",
	   learningUpdate = "reinforce",
	   model = "mlp"
	}
	local nIterations = 200

	local actionSpace = {}
	actionSpace.name = "Discrete"
	actionSpace.n = 6

	local stateSpace = {}
	stateSpace.name = "Box"
	stateSpace.shape = {}
	stateSpace.shape[1] = x
	stateSpace.shape[2] = y


	local agentOpt = {}
	agentOpt.stateSpace = stateSpace
	agentOpt.actionSpace = actionSpace
	agentOpt.nIterations = nIterations
	agentOpt.model = agent.model
	agentOpt.policy = agent.policy
	agentOpt.learningUpdate = agent.learningUpdate
	agentOpt.envDetails = util.getStateAndActionSpecs(stateSpace, actionSpace)


	local agent = require 'twrl.agent.baseAgent'(agentOpt)

	function predict(state)
		return agent.selectAction(nextState)
	end
	
	function reward(nIter, reward, terminal , nextstate ,nextaction , nIter)
		perf.addReward(nIter, reward, terminal)
		if action ~= nil then
			agent.reward({state = state, action = action, reward = reward, nextState = nextState, nextAction = nextAction, terminal = terminal, nIter = nIter})
		end
		state = nextstate
		action = nextaction
	end
	return {
		predict = predict,
		reward = reward
	}
end
return starcraft

