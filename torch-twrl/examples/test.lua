local sc = require 'twrl.starcraft'
a = sc(4,5,6)
print("*****")
print(sc)
print(a)

possibleState = {}
for i=1,4 do
	possibleState[i] = i
end
print(possibleState)
print(a.predict(possibleState))
print(a.predict(possibleState))
print(a.predict(possibleState))

