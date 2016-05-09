require 'nn'
require 'paths'
require 'image'

-- Download dataset
if (not paths.filep("cifar10torchsmall.zip")) then
	os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
	os.execute('unzip cifar10torchsmall.zip')
end
--Load the data and make classes
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset) 
print(#trainset.data)


setmetatable(trainset,
	{__index = function(t,i)
			return {t.data[i], t.label[i]}
		end}
);
trainset.data = trainset.data:double() 

function trainset:size()
	return self.data:size(1)
end

print(trainset[101])
itorch.image(trainset[101][1])

redChannel = trainset.data[{ {}, {1}, {}, {} }] -- picks all images, red channel, all x,y pixels
print(#redChannel)

--Set the mean to 0 and variance to 1 for all input
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean() -- mean estimate
	print('Channel ' .. i .. ', Mean: ' .. mean[i])
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i]) -- subtract off mean
	
	stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std() -- std estimate
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

--Implement a network to classify the images
net = nn.Sequential()

net:add(nn.SpatialConvolution(3,6,5,5)) ---3 input image channels, 6 output channels, 5x5 filter
net:add(nn.ReLU()) --f(x)= max(0,x)
net:add(nn.SpatialMaxPooling(2,2,2,2)) -- max pooling over 2x2 window

net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5)) -- reshape from 3D tensor of 16*5*5 into 1D tensor

net:add(nn.Linear(16*5*5,120)) -- fully connected layer (matrix multiplication between inputs and weights)
net:add(nn.ReLU())

net:add(nn.Linear(120,84)) -- full connected 120 inputs to 84
net:add(nn.ReLU())

net:add(nn.Linear(84,10))

net:add(nn.LogSoftMax()) -- coverts them to log probabilities 

criterion = nn.ClassNLLCriterion() -- Negative log likilihood loss criterion

trainer = nn.StochasticGradient(net, criterion) -- Create a stochastic gradient trainer
trainer.learningRate = .001
trainer.maxIteration = 5

trainer:train(trainset) -- train the network

testset.data = testset.data:double() -- convery from byte tensor to double tensor
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i]) -- substract mean from testing
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i]) -- div by stdv
end


print(classes[testset.label[100]])
itorch.image(testset.data[100]) -- We will check if this horse gets classifies by the nn matching this
predicted = net:forward(testset.data[100])

predicted = predicted:exp() -- the output of the network is log-likelihoods, so we exponentiate to get likilihoods.

for i=1,predicted:size(1) do -- Print the class labels with the probabilities
	print(classes[i],predicted[i])
end

correct = 0 --Check classification accuracy on testset
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction,true) -- true sorts in descending order
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end

print(correct, 100*correct/10000 .. ' % ') -- This gets around 40% accuracy. Let's see what it performs badly on 

class_performance = {0,0,0,0,0,0,0,0,0,0}
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)
	if groundtruth == indices[1] then
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end

for i=1,#classes do
	print(classes[i], 100*class_performance[i]/1000 .. ' % ')
end

--[[
net = net:cuda() -- transfer network to GPU
criterion = criterion:cuda() -- transfer loss function to GPU

trainset.data = trainset.data:cuda() -- trainset to GPU
trainset.label = trainset.label:cuda()

trainer = nn.SochasticGradient(net,criterion)
trainer.learningRate = .001
trainer.maxIteration = 5

trainer:train(trainset) -- Train the network on the GPU 
--]]
