local function getModel(opt)
   print(opt.nbLayers)
   local opt = opt or {}
   local nInputs = opt.nInputs
   local nOutputs = opt.nOutputs
   local nHiddenLayerSize = opt.nHiddenLayerSize
   local outputType = opt.outputType or 'categorical'
   local finalLayer = (outputType == 'categorical' or outputType == 'multicategorical') and nn.SoftMax()
      or nn.Tanh()
   local nbLayers = opt.nbLayers or 1
   local net = nn.Sequential()
      :add(nn.Linear(nInputs, nHiddenLayerSize))
      :add(nn.Tanh())
   for i=1,nbLayers do
      net:add(nn.Linear(nHiddenLayerSize, nHiddenLayerSize))
      :add(nn.Tanh())
   end
      net:add(nn.Linear(nHiddenLayerSize, nOutputs))
      :add(finalLayer)

   return {
      net = net 
   }
end

return getModel
