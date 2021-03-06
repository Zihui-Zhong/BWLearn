function get_centre_de_masse(unit_vector)
    i =0
    local centre_de_masse = {}
    centre_de_masse[1] = 0
    centre_de_masse[2] = 0
    for uid_enemy, ut_enemy in pairs(unit_vector) do
        centre_de_masse[1] = centre_de_masse[1] + ut_enemy.position[1]
	centre_de_masse[2] = centre_de_masse[2] + ut_enemy.position[2]
	i = i +1
    end
    centre_de_masse[1] = centre_de_masse[2] / i
    centre_de_masse[2] = centre_de_masse[1] / i
    return centre_de_masse
end	

function get_reward(friendly_units,hostile_units,troop_alive,action, lastReward, terminal)
    local reward = 0
    local index =0
    for i, is_alive in ipairs(troop_alive) do
        if (index >5) then
	    if (is_alive == 0) then
                reward = reward + 50
	     end
        elseif (is_alive == 1) then
		reward = reward + 50
	end
	index = index +1
    end

    for uid_friendly, ut_friendly in pairs(friendly_units) do
	reward = reward + ut_friendly.hp
    end
    for uid_enemy, ut_enemy in pairs(hostile_units) do
	reward = reward + (50 - ut_enemy.hp)
    end
    if (action == nil and terminal ~= true) then
        --reward = reward - 100
    end 
    return reward
end	

function flee_direction(position, centre_de_masse)
    -- calculate the flee normalized vector
    local new_pos = {position[1] - centre_de_masse[1], position[2] - centre_de_masse[2]}
    return new_pos
end
function get_command(ai_action, units_uid, is_alive_vector,tc) -- command can be from 1 to 29?
    local uids_index = math.floor(ai_action/6)+1
    local action = ai_action%6
    if (is_alive_vector[units_uid[uids_index]] == 0) then
        return nil
    end
    if (action == 5) then 
        local centre_de_masse = get_centre_de_masse(tc.state.units_enemy)
	local position = nil
	for uid, ut in pairs(tc.state.units_myself) do
	    if (uid == units_uid[uids_index]) then
		position = ut.position
	    end
	end
        position = flee_direction(position, centre_de_masse)
        return tc.command(tc.command_unit_protected, units_uid[uids_index],
                                   tc.cmd.Move, -1,
                                  position[1], position[2])
        
     else
	if (is_alive_vector[units_uid[action+6]] == 0) then
	    return nil
	end
        return tc.command(tc.command_unit_protected, units_uid[uids_index], tc.cmd.Attack_Unit, units_uid[6+action])
     end
end

--[[
   Copyright (c) 2015-present, Facebook, Inc.
   All rights reserved.
   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

-- attacks the closest units, th simple_exe.lua [-t $hostname] [-p $port]
local DEBUG = 0 -- can take values 0, 1, 2 (from no output to most verbose)
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'sys'
require 'gnuplot'
local lapp = require 'pl.lapp'
local args = lapp [[
Baselines for Starcraft
    -t,--hostname       (default "")    Give hostname / ip pointing to VM
    -p,--port           (default 11111) Port for torchcraft. Do 0 to grab vm from cloud
]]

local lapp = require 'pl.lapp'
local args = lapp [[
Baselines for Starcraft
    -t,--hostname       (default "")    Give hostname / ip pointing to VM
    -p,--port           (default 11111) Port for torchcraft. Do 0 to grab vm from cloud
]]

local skip_frames = 4
local port = args.port
local hostname = args.hostname or ""
print("hostname:", hostname)
print("port:", port)

require 'progressbar'
local progress = ProgressBar(-1)
progress.mutex = {lock = function() end, unlock = function() end}
progress:singleThreaded()

local tc = require 'torchcraft'
tc.DEBUG = DEBUG
local utils = require 'torchcraft.utils'

local function get_closest(position, unitsTable)
    local min_d = 1E30
    local closest_uid = nil
    for uid, ut in pairs(unitsTable) do
        local tmp_d = utils.distance(position, ut['position'])
        if tmp_d < min_d then
            min_d = tmp_d
            closest_uid = uid
        end
    end
    return closest_uid
end

local battles_won = 0
local battles_game = 0
local total_battles = 0

local frames_in_battle = 1
local nloop = 1

-- no ',' accepted in map names!
-- All paths must be relative to C:/StarCraft
local maps = {'Maps/BroodWar/micro/m5v5_c_far.scm'}

tc.micro_battles = true
-- This overwrites whatever is in bwapi.ini
tc.initial_map = maps[2]

-- connects to the StarCraft launched by BWEnv.exe
tc:init(hostname, port)
local update = tc:connect(port)

if DEBUG > 1 then
    print('Received init: ', update)
end
assert(tc.state.replay == false)

-- first message to BWAPI's side is setting up variables
local setup = {
    tc.command(tc.set_speed, 0), tc.command(tc.set_gui, 1),
    tc.command(tc.set_frameskip, 400),
    tc.command(tc.set_cmd_optim, 1),
}
tc:send({table.concat(setup, ':')})


local nrestarts = -1
--machine learning intialization part

local sc = require 'twrl.starcraft'

opt = {}
opt.nbInput = 4   --table size
opt.nbAgent = 5   -- nb unit to control
opt.nbActions = 6 -- nb action per unit
opt.neuronesPerLayer= 200
opt.nbHiddenLayer = 5
opt.networktype = ""
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
opt.tdLearnUpdate = "SARSA"
--opt.pathToOldNetwork = "oldModel"
local model = sc(opt)

local score = {}
local winRatio = {}
local maxBattlesBeforeRestart = 200
while total_battles < 40000 do

    print("")
    print("CTRL-C to stop")
    print("")

    battles_won = 0
    battles_game = 0
    nrestarts = nrestarts + 1

    -- initialize TorchCraft's state
    tc:set_variables()

    local tm = torch.Timer()
    local gameTimer = os.time()
    local taking_uid = true
    local compute_reward = false
    local uids = {}
    local troop_action = {}
    local troop_alive = {}
    local lastAction = nil
    local lastState = nil
    local lastScore = 0
    local failed = false
    while not tc.state.game_ended do
        local done = false
        update = tc:receive()
        if DEBUG > 1 then
            print('Received update: ', update)
        end
        nloop = nloop + 1
	local actions = {}
	
        if tc.state.game_ended then
            done = true
        elseif tc.state.battle_just_ended then
            if DEBUG > 0 then
                print("BATTLE ENDED")
            end
	    if (failed) then
		model.reward(units,action, total_battles, -500, true , lastState ,lastAction )
	        --table.insert(score,-500)
            else
		model.reward(units,action, total_battles, get_reward(tc.state.units_myself,tc.state.units_enemy, troop_alive, true), true , lastState ,lastAction )
	        table.insert(score,get_reward(tc.state.units_myself,tc.state.units_enemy, troop_alive,true))
	    end
	    taking_uid = true
    	    compute_reward = false
	    lastScore = 0
	    failed = false
    	    uids = {}
    	    troop_action = {}
    	    troop_alive = {}
            lastAction = nil
            lastState = nil
	    gameTimer = os.time()
            if tc.state.battle_won then -- we won (in micro battles)
                battles_won = battles_won + 1
            end
            battles_game = battles_game + 1
            total_battles = total_battles + 1
            frames_in_battle = 0
            if battles_game >= maxBattlesBeforeRestart then -- this is an example
                done = true
            end
        elseif tc.state.waiting_for_restart then
            -- a battle finished, waiting for the next one to start!
            if DEBUG > 0 then
                print("WAITING FOR RESTART")
            end
        else
            if tc.state.battle_frame_count % skip_frames == 0 then
		if taking_uid then
		    for uid_ally, ut_ally in pairs(tc.state.units_myself) do
			table.insert(uids,uid_ally)
			troop_action[uid_ally] = 999
			troop_alive[uid_ally] = 1
			--print("trying to print UID")
			--print(uid_ally)
		    	--table.insert(units,ut.health)
			--print("Trying to print UT")
		    	--print(ut_ally)
		    end 
		    for uid_enemy, ut_enemy in pairs(tc.state.units_enemy) do
			table.insert(uids,uid_enemy)
			troop_action[uid_enemy] = 999
			troop_alive[uid_enemy] = 1
		    end
		    --print(uids) 
		    taking_uid = false
		end
                units = {}
		for i, uid in ipairs(uids) do
			local thisUnit = tc.state.units_myself[uid]
			if thisUnit == nil then
				thisUnit = tc.state.units_enemy[uid]
			end
			if thisUnit == nil then
				table.insert(units,0) -- hp to 0
				table.insert(units,999) -- action = 999 when dead
				table.insert(units,0) -- unit is dead
				troop_alive[uid] = 0
				for j, uid2 in ipairs(uids) do 
					table.insert(units,999) -- distance
				end
			else
				table.insert(units,thisUnit.hp)
				--table.insert(units,thisUnit.shield)
				table.insert(units,troop_action[uid]) -- inserting last troop action
				table.insert(units,1) -- troop is alive
				troop_alive[uid] = 1
				for j, second_pass_uid in ipairs(uids) do
					local thatUnit = tc.state.units_myself[second_pass_uid]
					if thatUnit == nil then
						thatUnit = tc.state.units_enemy[second_pass_uid]
					end 
					if thatUnit == nil then
						table.insert(units,999)
					else
						local x = thatUnit.position[1] - thisUnit.position[1]
						local y = thatUnit.position[2] - thisUnit.position[2]
						local dist = x*x + y*y
						table.insert(units,dist)  
					end
				end
				
			end
		
		end
		-- if the AI get stuck in a stupid pattern we override the command stop the learning and go suicide so we can start back from scratch
		if ((os.time() - gameTimer) > 8 ) then
			--call the actions shenenigan here
		        for uid, ut in pairs(tc.state.units_myself) do
			    --print(ut)
		            local target = get_closest(ut.position,
		                                       tc.state.units_enemy)
		   	    --print(target)
		            if target ~= nil then
		                table.insert(actions,
		                    tc.command(tc.command_unit_protected, uid,tc.cmd.Attack_Unit, target))
		            end
		        end
			failed = true
		else
		
			local action = model.predict(units)
			local score = 0
			if (compute_reward) then
			    score = get_reward(tc.state.units_myself,tc.state.units_enemy, troop_alive, false)  
			    model.reward(units,action, total_battles, score, false , lastState ,lastAction, false )
		        end
		        compute_reward = true
			lastAction = action
			lastState = units
			lastScore = score
			local command = get_command(action, uids, troop_alive,tc)
			table.insert(actions,command)
		end
                --if frames_in_battle > 2*60*24 then -- quit after ~ 2 hours
                    --actions = {tc.command(tc.quit)}
                    --nrestarts = nrestarts + 1
                --end
                progress:pop()
            end
        end

        progress:add('Loop', nloop, '%5d')
        progress:add('FPS', 1 / tm:time().real, '%5d')
        progress:add('WR', battles_won / (battles_game+1E-6), '%1.3f')
        progress:add('#Wins', battles_won, '%4d')
        progress:add('#Bttls', battles_game, '%4d')
        progress:push()
        tm:reset()
        if done then
	    table.insert(winRatio,battles_won / (battles_game+1E-6))
	    break 
        end -- So the last progress bar update happens

        if DEBUG > 1 then
            print("")
            print("Sending actions:")
            print(actions)
        end
        tc:send({table.concat(actions, ':')})
    end

    local m = maps[(nrestarts % #maps) + 1]
    local setup = {
        tc.command(tc.set_map, m),
        tc.command(tc.restart),
        tc.command(tc.set_speed, 0), tc.command(tc.set_gui, 1),
        tc.command(tc.set_frameskip, 400),
        tc.command(tc.set_cmd_optim, 1),
    }
    print("")
    print("Current map name: ", tc.state.map_name)
    print("Sent map name: ", m)

    tc:send({table.concat(setup, ':')})
    os.execute("sleep 0.5")
    print("")
    progress:reset()
    print("")
    collectgarbage()
    collectgarbage()
end
file = io.open("score.txt", "a")
io.output(file)
for i, number in ipairs(score) do
    io.write(number, "\n")

end
io.close(file)
file = io.open("winRatio.txt", "a")
io.output(file)
for i, number in ipairs(winRatio) do
    io.write(number, "\n")

end
io.close(file)
tc:receive()
tc:send({table.concat({tc.command(tc.exit_process)})})
