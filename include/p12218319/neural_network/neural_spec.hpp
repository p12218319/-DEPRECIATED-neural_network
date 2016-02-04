#ifndef P12218319_NEURAL_NETWORK_NERUAL_SPEC_HPP
#define P12218319_NEURAL_NETWORK_NERUAL_SPEC_HPP

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

namespace p12218319 { namespace neural_network {

    /*!
        \class neural_spec
        \brief Defines the input and output type helpers for a neuron
        \tparam INPUT_TYPE The input type
        \tparam WEIGHT_TYPE The weight type
        \tparam INPUTS The number of inputs
        \version 1.0
    */
    template<class INPUT_TYPE, class WEIGHT_TYPE, const int INPUTS>
    struct neural_spec {
        enum {
            input_count = INPUTS                                        //!< The number of inputs for the neuron
        };
        typedef INPUT_TYPE input_type;                                  //!< The type of the neuron's inputs
        typedef INPUT_TYPE output_type;                                 //!< The type of the neurons's output
        typedef WEIGHT_TYPE weight_type;                                //!< The type of the neuron's weights
        typedef input_type input_array[input_count];                    //!< An array that can contain the inputs to the neuron
        typedef const input_type const_input_array[input_count];        //!< A read-only array that can contain the inputs to the neuron
        typedef weight_type weight_array[input_count];                  //!< An array that can contain the weights for the neuron
        typedef const weight_type const_weight_array[input_count];      //!< A read-only array that can contain the weights for the neuron

        typedef input_type (*weighted_sum)(const_weight_array&, const_input_array&);        //!< A function that can perform a weight sum of inputs
        typedef output_type(*activation_function)(const input_type);                        //!< A function that can perform the activation function of the neuron

        static constexpr weight_type default_weight = static_cast<WEIGHT_TYPE>(1) / static_cast<WEIGHT_TYPE>(INPUTS);   //!< A weight value that gives equal weighting to all inputs

		/*!
			\brief The default sum behaviour for neurons, multiplies each input by it's matching weight and sums the results
			\param aWeights An array containing the weighting values
			\param aInputs An array containing the input values
			\return The weighted sum
		*/
        static input_type default_sum(const_weight_array& aWeights, const_input_array& aInputs) throw() {
            input_type sum = static_cast<input_type>(0);
            for(int i = 0; i < input_count; ++i) sum += aInputs[i] * static_cast<const input_type>(aWeights[i]);
            return sum;
        }

		/*!
			\brief The default activation function for neurons, passes the weighted sum directly to the next neuron
			\param aSum The output of the weighted sum
			\return Equal to \a aSum
		*/
        static output_type pass_through(const input_type aSum) throw() {
            return static_cast<output_type>(aSum);
        }
    };

}}

#endif
