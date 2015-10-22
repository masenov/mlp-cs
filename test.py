# include here the complete code that constructs the model, performs training,
# and prints the error and accuracy for train/valid/test
import numpy


from mlp.layers import MLP, Linear, Sigmoid, Softmax #import required layer types
from mlp.optimisers import SGDOptimiser #import the optimiser
from mlp.dataset import MNISTDataProvider #import data provider
from mlp.costs import CECost, MSECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed

rng = numpy.random.RandomState([2015,10,10])

# define the model structure, here just one linear layer
# and mean square error cost
cost = CECost()
model = MLP(cost=cost)
#model.add_layer(Linear(idim=784, odim=100, rng=rng))
#model.add_layer(Linear(idim=100, odim=10, rng=rng))
model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
model.add_layer(Softmax(idim=100, odim=10, rng=rng))
#one can stack more layers here

# define the optimiser, here stochasitc gradient descent
# with fixed learning rate and max_epochs as stopping criterion
lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=20)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

train_dp = MNISTDataProvider(dset='train', batch_size=99, max_num_batches=-10, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=99, max_num_batches=-10, randomize=False)

optimiser.train(model, train_dp, valid_dp)

test_dp = MNISTDataProvider(dset='eval', batch_size=99, max_num_batches=-10, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
print "train"