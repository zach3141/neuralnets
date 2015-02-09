__author__ = 'zbutler'
import numpy as np

def backprop_1iter(net, inputs, targets, alpha=.05):
    """ Run one step of the backpropagation algorithm
    :param net: A neural network of the NN class
    :param inputs: A numpy array of input vectors, one vector per row
    :param targets: A numpy array of targets in the same format and order of inputs
    :return: Nothing, but updates net to better fit the input vectors
    """
    npts = inputs.dim[0]
    if npts != targets.dim[0]:
        raise IndexError("Number of targets needs to match number of inputs")

    # First, forward propagate all of our inputs to get network outputs
    outputs = np.array(targets.dim,dtype=targets.float)
    for pt in xrange(npts):
        from_previous = inputs[pt,:]
        for next_layer in net.W:
            from_previous = next_layer * np.transpose(from_previous)
        outputs[pt,:] = from_previous

    # Now, compute error and backpropagate:
    for pt in xrange(npts):
        tar = targets[pt,:]
        out = outputs[pt,:]
        err = net.erf(tar,out)
        deltas =

