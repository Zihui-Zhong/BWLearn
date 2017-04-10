local function categorical(actionDistribution, opt)
   local opt = opt or {}
   local actionShift = opt.actionShift
   
   return (torch.multinomial(actionDistribution, 1) - actionShift)[1][1]
end


local function normal(actionDistribution, opt)
   local opt = opt or {}
   local std = opt.std
   local actionBoundFactor = opt.actionBoundFactor
   local actions = torch.cmul(torch.normal(actionDistribution, std), actionBoundFactor)
   actions = actions:size():size() == 2 and actions[1] or actions
   return actions:totable()
end

local function multicategorical(actionDistribution, opt)

   local nbAgent = opt.nbAgent
   local nbRows = opt.envDetails.nbActions/nbAgent
   local temp = torch.multinomial(actionDistribution:resize(nbAgent,nbRows),1)
   ret = {}
   local actionShift = opt.actionShift
   for i=1,nbAgent do
      ret[i] = temp[i][1]-actionShift
   end
   print(ret)


   
   return ret
end

return {
   categorical = categorical,
   multicategorical = multicategorical,
   normal = normal
}
