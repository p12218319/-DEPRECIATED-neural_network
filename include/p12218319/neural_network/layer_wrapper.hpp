#ifndef P12218319_NEURAL_NETWORK_LAYER_WRAPPER_HPP
#define P12218319_NEURAL_NETWORK_LAYER_WRAPPER_HPP

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

#include <vector>
#include "neuron_wrapper.hpp"
#include "layer.hpp"

namespace p12218319 { namespace neural_network {

	/*!
		\class layer_wrapper
		\brief A wrapper class that allows runtime access to a network layer
		\tparam INPUT_TYPE The type of the neurons' input and outputs
		\tparam WEIGHT_TYPE The type of the neurons' weighting values
		\see layer
		\see neuron_wrapper
	*/
    template<class INPUT_TYPE, class WEIGHT_TYPE>
    class layer_wrapper {
    private:
        std::vector<neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>> mNeurons;	//!< A list of neuron_wrapper objects that make up the layer
    public:
		/*!
			\class layer_wrapper
			\brief A wrapper that allows access to the layer's neurons with the same semantics as layer::neurons
			\see layer::neurons
		*/
        class neuron_array{
        private:
            layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>& mLayer;	//!< The layer_wrapper that owns this array
        public:

            // Functions

			/*!
				\brief Get the number of neurons in the layer
				\return The neuron count
			*/
            inline int size() const throw() {
                return mLayer.mNeurons.size;
            }

            // Operators

			/*!
				\brief Access a neuron in the layer
				\param aIndex The index of the neuron to get
				\return A reference to the neuron
			*/
            inline neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>& operator[](const int aIndex) throw() {
                return mLayer.mNeurons[aIndex];
            }

			/*!
				\brief Access a neuron in the layer
				\param aIndex The index of the neuron to get
				\return A reference to the neuron
			*/
            inline const neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>& operator[](const int aIndex) const throw() {
                return mLayer.mNeurons[aIndex];
            }

            // Constructors


			/*!
				\brief Create a new layer_wrapper that defers calls to the neurons in an existing layer object
				\param aLayer The layer to create neuron_wrapper(s) for
			*/
            neuron_array(layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>& aLayer) throw() :
                mLayer(aLayer)
            {}
        };

        neuron_array neurons;	//!< A wrapper that allows access to the layer's neurons with the same semantics as layer::neurons
    public:
        // Operators

		/*!
			\brief Call each neuron in the layer with the same inputs
			\param aInputs The input values to pass to each neuron
			\param aOutputs Holds the output of each neuron
			\see layer::operator()
		*/
        void operator()(const INPUT_TYPE* const aInputs, INPUT_TYPE* const aOutputs) const throw() {
            const int size = mNeurons.size();
            for(int i = 0; i < size; ++i) aOutputs[i] = mNeurons[i](aInputs);
        }

        // Constructors

		/*!
			\brief Create a new layer_wrapper
			\tparam SPEC The neural_spec class that defines the neurons in \a aLayer, this is automatically determined by the compiler
			\tparam NEURON_COUNT The number of neurons in \a aLayer, this is automatically determined by the compiler
			\param aLayer The layer to defer neuron calls to
		*/
        template<class SPEC, const int NEURON_COUNT>
        layer_wrapper(layer<SPEC, NEURON_COUNT>& aLayer) :
            neurons(*this)
        {
            for(int i = 0; i < NEURON_COUNT; ++i) mNeurons.push_back(neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>(aLayer.neurons[i]));
        }
    };

}}

#endif
