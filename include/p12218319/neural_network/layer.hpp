#ifndef P12218319_NEURAL_NETWORK_LAYER_HPP
#define P12218319_NEURAL_NETWORK_LAYER_HPP

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

#include "neural_spec.hpp"
#include "neuron.hpp"

namespace p12218319 { namespace neural_network {

	/*!
		\class layer
		\brief A group of related neurons that share inputs, can be used as a layer of a neural network
		\tparam SPEC The neural_spec class that defines the neurons in this layer
		\tparam NEURON_COUNT The number of neurons in this layer
		\version 1.0
		\see neuron
	*/
    template<class SPEC, const int NEURON_COUNT>
    struct layer {
        enum {
            output_count = NEURON_COUNT		//!< The number of neurons in the layer, and the number of outputs from this layer to the next
        };
        typedef SPEC spec;                                                                                      //!< The neural_spec class for the neurons in this layer
        typedef neural_spec<typename spec::input_type, typename spec::weight_type, output_count> next_spec;     //!< The neural_spec class for the next layer of the network
        typedef typename spec::output_type output_array[output_count];                                          //!< An array that can contain the outputs of this layer
        typedef const typename spec::output_type const_output_array[output_count];                              //!< A read-only array that can contain the outputs of this layer

        neuron<spec> neurons[output_count];     //!< The neurons in this layer

        // Operators

		/*!
			\brief Call each neuron in the layer
			\param aInputs The inputs to pass to each neuron
			\param aOutputs An array that will store the output of each neuron
			\see neuron::operator()
		*/
        void operator()(typename spec::const_input_array& aInputs, output_array& aOutputs) const throw() {
            for(int i = 0; i < output_count; ++i) aOutputs[i] = neurons[i](aInputs);
        }

        // Constructors
    };

}}

#endif
