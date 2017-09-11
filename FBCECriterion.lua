-- @FBCECriterion: focal loss BCE 
-- @Date:   2017-09-07 14:46:50
-- @Last Modified by:   18mat
-- @Last Modified time: 2017-09-08 18:18:36
local FBCECriterion, Parent = torch.class('nn.FBCECriterion', 'nn.Criterion')
local eps = 1e-12
function FBCECriterion:__init(gamma, alpha, sizeAverage)
	Parent.__init(self)
	if gamma ~= nil then
    	self.gamma = gamma
   	else
    	self.gamma = 0
   	end
   	if alpha ~= nil then
    	self.alpha = alpha
   	else
    	alpha = 0.5
    	self.alpha = alpha
   	end
   	if sizeAverage ~= nil then
    	self.sizeAverage = sizeAverage
   	else
    	self.sizeAverage = true
   	end

    -- if isinstance(alpha,(float,int,long)) then
    -- 	self.alpha = torch.Tensor([alpha,1-alpha])
    -- if isinstance(alpha,list) then
    -- 	self.alpha = torch.Tensor(alpha)
    -- 	self.size_average = size_average
end

function FBCECriterion:updateOutput(input, target)
	-- print(self.gamma)

	assert( input:nElement() == target:nElement(), "input and target size mismatch")
	self.output_tensor = self.output_tensor or input.new(1)
	-- cls_score = in_data[0].asnumpy()
	-- labels = in_data[1].asnumpy()
	-- self._labels = labels

	self.term1 = self.term1 or input.new()
	self.term2 = self.term2 or input.new()
	self.term3 = self.term3 or input.new()
	self.term4 = self.term4 or input.new()
	self.term5 = self.term5 or input.new()

	self.term1:resizeAs(input)
	self.term2:resizeAs(input)
	self.term3:resizeAs(input)
	self.term4:resizeAs(input)
	self.term5:resizeAs(input)

	self.term1:fill(1):add(-1,target)
	self.term5:fill(1):add(-1,input):pow(self.gamma)
	self.term4:copy(input):pow(self.gamma)


	self.term2:fill(1):add(-1,input):add(eps):log():cmul(self.term1):cmul(self.term4)	
	self.term3:copy(input):log():cmul(target):cmul(self.term5)
	self.term3:add(self.term2)

	-- 
	-- self.term3:cmul(self.term4)

	if self.sizeAverage then
		self.term3:div(target:nElement())
	end

	self.output = - self.term3:sum()

	return self.output
end

function FBCECriterion:updateGradInput(input, target)

	self.term1 = self.term1 or input.new()
	self.term2 = self.term2 or input.new()
	self.term3 = self.term3 or input.new()
	self.term4 = self.term4 or input.new()
	self.term5 = self.term5 or input.new()
	self.term6 = self.term6 or input.new()
	self.term7 = self.term7 or input.new()
	self.term8 = self.term8 or input.new()

	self.term1:resizeAs(input)
	self.term2:resizeAs(input)
	self.term3:resizeAs(input)
	self.term4:resizeAs(input)
	self.term5:resizeAs(input)
	self.term6:resizeAs(input)
	self.term7:resizeAs(input)
	self.term8:resizeAs(input)

	self.term1:fill(1):add(-1, input)
	self.term2:fill(1):add(-1, target)

	self.term3:copy(self.term1):cdiv(input)
	self.term4:copy(input):log():mul(self.gamma)
	self.term3:add(-1, self.term4)
	self.term5:copy(self.term1):pow(self.gamma-1):cmul(target):cmul(self.term3)

	self.term1:add(eps)
	self.term6:copy(input):cdiv(self.term1)
	self.term7:copy(self.term1):log():mul(self.gamma)
	self.term6:add(-1,self.term7)
	self.term8:copy(input):pow(self.gamma-1):cmul(self.term2):cmul(self.term6)

	-- self.term1:copy(input):log():mul(self.gamma)

	-- self.term5:copy(self.term4):log():mul(self.gamma)
	-- self.term4:add(eps):pow(-1):add(self.term5)

	-- self.term2:cmul(self.term3):cmul(self.term4)


	self.gradInput:resizeAs(input)
	self.gradInput:copy(self.term5):add(-1, self.term8)   


	if self.sizeAverage then
	  self.gradInput:div(target:nElement())
	end

	self.gradInput:mul(-1)

	return self.gradInput
end

-- function FocalBCECriterion:accGradParameters(input, gradOutput)
-- end
