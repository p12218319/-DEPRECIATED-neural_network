#ifndef P12218319_NEURAL_NETWORK_DOXYGEN_HPP
#define P12218319_NEURAL_NETWORK_DOXYGEN_HPP

//Copyright 2015 P12218319 - Adam Smith
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

// Contact -
// Email   	: p12218319@myemail.dmu.ac.uk
// Github	: https://github.com/p12218319/neural_network

/*!
    \mainpage P12218319 Neural Network

    \section neuron_introduction Neuron class introduction

    Before a neuron can be created, a neural_spec class must be defined.
    This defines the input, output and weighting types, as well as how many inputs the neuron will have.
    Using a typedef for this is convenient:
    \code
    // We will define a neuron that has two inputs, and uses floats for all calculations
    typedef neural_spec<float, float, 2> neuron_spec_2f;

    // We can now use the neural_spec class to create a neuron
    typedef neuron<neuron_spec_2f> neuron_2f; // Typedef the neuron class so it is easier to refer to later
    neuron_2f myNeuron;
    \endcode
    The neuron class has three programmable components, the weighing values, the weighting sum, and the activation function.
    The default weighting values and sum functionally take an average of the inputs.
    The default activation function is a pass-through for the weighting sum.
    \code
    // Programming the weighting values
    myNeuron.weights[0] = 0.5f;
    myNeuron.weights[1] = 0.5f;

    // Programming the weighting values
    myNeuron.weights[0] = 0.5f;
    myNeuron.weights[1] = 0.5f;

    // Programming the weighting sum
    // For this example, we will duplicate the behaviour of the default sum
    myNeuron.weighted_sum = [](neuron_spec_2f::const_weight_array& aWeights, neuron_spec_2f::const_input_array& aInputs)->float{
        // Multiply the inputs by the weighting values and add them
        return (aInputs[0] * aWeights[0]) + (aInputs[1] * aWeights[1]);
    };

    // Programming the activation
    // For this example, the neuron will be activated by a simple stepping function
    myNeuron.activation_function = [](const float aSum)->float{
        return aSum < 0.f ? 0.f : 1.f;
    };
    \endcode

    \section layer_introduction Layer class introduction
    Todo
	
    \section network_introduction Network class introduction
    Todo
	
    \section generic_introduction Using a network without templated code
    With the classes 
*/

#endif
