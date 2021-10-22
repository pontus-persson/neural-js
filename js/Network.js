var Network = function(topology) {
    var self = this;

    this.layers = [];
    this.outputLayer = 0;

    this.error = 0;
    this.recentAverageError = 0;
    this.recentAverageErrorSmoothing = 50.0;

    for (var l = 0; l < topology.length; l++) {
        this.layers[l] = new Layer(topology[l], topology[l+1]);
        this.outputLayer = l;
    }

    this.feedForward = function(data) {
        for (var x = 0; x < data.length; x++) {
            for (var y = 0; y < data[x].length; y++) {
                this.layers[0].neurons[x][y].value = data[x][y];
            }
        }

        for (var l = 1; l < this.layers.length; l++) {
            this.layers[l].feedForward(this.layers[l-1]);
        }
    }


    this.backPropagation = function(targetValues) {

        this.error = 0.0;

        // Calculate overall net error
        for (var x = 0; x < this.layers[this.outputLayer].neurons.length; x++) {
            for (var y = 0; y < this.layers[this.outputLayer].neurons[x].length; y++) {
                var delta = targetValues[x][y] - this.layers[this.outputLayer].neurons[x][y].value;
                this.error += delta * delta;
                // Calculate output layer gradients
                this.layers[this.outputLayer].neurons[x][y].calculateOutputGradients(targetValues[x][y]);
            }
        }

        this.error /= this.layers[this.outputLayer].totalNeurons;
        this.error = Math.sqrt(this.error);
        
        this.recentAverageError = 
            (this.recentAverageError * this.recentAverageErrorSmoothing + this.error) / 
            (this.recentAverageErrorSmoothing + 1.0);

        // Calclate gradients on hidden layers
        for (var l = this.layers.length - 2; l > 0; l--) {
            for (var x = 0; x < this.layers[l].neurons.length; x++) {
                for (var y = 0; y < this.layers[l].neurons[x].length; y++) {
                    this.layers[l].neurons[x][y].calculateHiddenGradients(this.layers[l + 1]);
                }
            }
        }

        // Update connection weights: for all layers from output to first hidden layer
        for (var l = this.layers.length - 1; l > 0; l--) {
            for (var x = 0; x < this.layers[l].neurons.length; x++) {
                for (var y = 0; y < this.layers[l].neurons[x].length; y++) {
                    this.layers[l].neurons[x][y].updateInputWeights(this.layers[l - 1]);
                }
            }
        }

        // console.log(this.error);
        // console.log(this.recentAverageError);
    }

    this.getReadableResult = function() {
        var highest = 0;
        var sum = 0;
        var result = 0;

        for (var x = 0; x < this.layers[this.outputLayer].neurons.length; x++) {
            for (var y = 0; y < this.layers[this.outputLayer].neurons[x].length; y++) {
                sum += Math.abs(this.layers[this.outputLayer].neurons[x][y].value);
                if(highest < this.layers[this.outputLayer].neurons[x][y].value) {
                    result = x;
                    highest = this.layers[this.outputLayer].neurons[x][y].value;
                }
            }
        }

        var percent = (highest / sum) * 100;

        return 'I belive that is a <span class="big">'+result+'</span> '+percent+'% sure...';
    }

}

var Layer = function(thisLayer, nextLayer) {
    var self = this;

    this.totalNeurons = 0;
    this.neurons = [];
    for (var x = 0; x < thisLayer.size.x; x++) {
        this.neurons[x] = [];
        for (var y = 0; y < thisLayer.size.y; y++) {
            this.neurons[x][y] = new Neuron(x, y, thisLayer, nextLayer);
            this.totalNeurons++;
        }
    }

    this.feedForward = function(prevLayer) {
        for (var x = 0; x < this.neurons.length; x++) {
            for (var y = 0; y < this.neurons[x].length; y++) {
                this.neurons[x][y].feedForward(prevLayer);
            }
        }
    }
}

var Neuron = function(posx, posy, thisLayer, nextLayer) {
    var self = this;

    this.gradient = 0;
    this.value = Math.random();
    this.position = { x:posx, y:posy };
    this.weights = [];

    // only add weights if we have a next layer
    if (typeof nextLayer !== 'undefined') {
        var offset = {
            x: Math.floor(nextLayer.size.x / 2) - Math.floor(thisLayer.size.x / 2),
            y: Math.floor(nextLayer.size.y / 2) - Math.floor(thisLayer.size.y / 2),
        }

        if(thisLayer.radius === true) {
            console.log("true radius");
            for (var x = 0; x < nextLayer.size.x; x++) {
                for (var y = 0; y < nextLayer.size.y; y++) {
                    this.weights.push(new Connection(x, y));
                }
            }
        } else if(thisLayer.radius > 0) {
            console.log("pos radius");
            var targetX, targetY;
            for (var x = -thisLayer.radius; x <= thisLayer.radius; x++) {
                for (var y = -thisLayer.radius; y <= thisLayer.radius; y++) {
                    targetX = this.position.x + offset.x + x;
                    targetY = this.position.y + offset.y + y;
                    if( targetX >= 0 && targetX < nextLayer.size.x &&
                        targetY >= 0 && targetY < nextLayer.size.y) {
                        this.weights.push(new Connection(targetX, targetY));
                    }
                }
            }
        } else if(thisLayer.radius < 0) {
            console.log("neg radius");
            var rad = Math.abs(thisLayer.radius);
            var targetX = Math.floor(this.position.x / rad);
            var targetY = Math.floor(this.position.y / rad);
            this.weights.push(new Connection(targetX, targetY));
        } else if(thisLayer.radius === 0) {
            console.log("zero radius");
        }
    }

    /**
     * Takes all neurons from last layer with a connection to this neuron 
     * and calculates the new outputvalue
     */
    this.feedForward = function(prevLayer) {
        var sum = 0;
        for (var x = 0; x < prevLayer.neurons.length; x++) {
            for (var y = 0; y < prevLayer.neurons[x].length; y++) {
                for (var w = 0; w < prevLayer.neurons[x][y].weights.length; w++) {
                    if( prevLayer.neurons[x][y].weights[w].x == this.position.x &&
                        prevLayer.neurons[x][y].weights[w].y == this.position.y) {
                        sum += prevLayer.neurons[x][y].value * prevLayer.neurons[x][y].weights[w].weight;
                    }
                }
            }
        }
        this.value = Math.tanh(sum);
    }

    this.calculateOutputGradients = function(targetValue) {
        var delta = targetValue - this.value;
        this.gradient = delta * (1.0 - this.value * this.value);
    }

    this.calculateHiddenGradients = function(nextLayer) {
        var sumDOW = 0.0;
        for (var x = 0; x < nextLayer.neurons.length; x++) {
            for (var y = 0; y < nextLayer.neurons[x].length; y++) {
                for (var w = 0; w < nextLayer.neurons[x][y].weights.length; w++) {
                    if (nextLayer.neurons[x][y].weights[w].x == x && nextLayer.neurons[x][y].weights[w].y == y) {
                        // console.log(nextLayer.neurons[x][y].weights[w].weight);
                        // console.log(nextLayer.neurons[x][y].gradient);
                        sumDOW += nextLayer.neurons[x][y].weights[w].weight * nextLayer.neurons[x][y].gradient;
                    }
                }
            }
        }
        // console.log(sumDOW);
        this.gradient = sumDOW * (1.0 - this.value * this.value);
    }


    this.updateInputWeights = function(prevLayer) {
        for (var x = 0; x < prevLayer.neurons.length; x++) {
            for (var y = 0; y < prevLayer.neurons[x].length; y++) {
                for (var w = 0; w < prevLayer.neurons[x][y].weights.length; w++) {
                    if (prevLayer.neurons[x][y].weights[w].x == this.position.x &&
                        prevLayer.neurons[x][y].weights[w].y == this.position.y)
                    {
                        var oldDeltaWeight = prevLayer.neurons[x][y].weights[w].deltaWeight;
                        var newDeltaWeight =
                            0.05 *                  // ETA, overall learning rate
                            prevLayer.neurons[x][y].value * //
                            this.gradient +         //
                            (0.4 * oldDeltaWeight); // keep a fraction of old deltaweight

                        prevLayer.neurons[x][y].weights[w].deltaWeight = newDeltaWeight;
                        prevLayer.neurons[x][y].weights[w].weight += newDeltaWeight;
                    }
                }
            }
        }
    }

}


var Connection = function(tx, ty) {
    var self = this;
    this.x = tx;
    this.y = ty;
    this.weight =      Math.random() * 2 - 1;
    this.deltaWeight = Math.random() * 2 - 1;
}