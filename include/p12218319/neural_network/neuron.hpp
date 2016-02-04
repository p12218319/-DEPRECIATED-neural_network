#ifndef P12218319_NEURAL_NETWORK_NEURON_HPP
#define P12218319_NEURAL_NETWORK_NEURON_HPP

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
        \class neuron
        \brief Represents a neuron with a fixed number of inputs, but with programmable weighting and activation
        \tparam SPEC The neural_spec class that defines this neuron
        \version 1.0
        \see neural_spec
    */
    template<class SPEC>
    struct neuron {
        typedef SPEC spec;  //!< The neural_spec class for this neuron

        typename spec::weight_array weights;                    //!< Contains the input weightings for this neuron
        typename spec::weighted_sum weighted_sum;               //!< Determines the weighted sum output for this neuron
        typename spec::activation_function activation_function; //!< Determines the activation function output for this neuron

        // Operators

        /*!
            \brief Pass inputs to the neuron and retrieve the output
            \param aInputs The inputs to the neuron
            \return The output of the neuron
        */
        inline typename spec::output_type operator()(typename spec::const_input_array& aInputs) const throw() {
            return activation_function(weighted_sum(weights, aInputs));
        }

        // Constructors

        /*!
            \brief Create a pass-through neuron with equal weighting of all inputs
        */
        neuron() :
            weighted_sum(spec::default_sum),
            activation_function(spec::pass_through)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = spec::default_weight;
        }

        /*!
            \brief Create a pass-through neuron with specified weighting of all inputs
            \param aWeight The input weight for all inputs
        */
        neuron(const typename spec::weight_type aWeight) :
            weighted_sum(spec::default_sum),
            activation_function(spec::pass_through)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = aWeight;
        }

        /*!
            \brief Create a pass-through neuron with specified weighting of inputs
            \param aWeights The input weightings for this neuron
        */
        neuron(typename spec::const_weight_array& aWeights) :
            weights(aWeights),
            weighted_sum(spec::default_sum),
            activation_function(spec::pass_through)
        {}

        /*!
            \brief Create a specified activation neuron with equal weighting of all inputs
            \param aActivationFunction The function that controls the activation behaviour of this neuron
        */
        neuron(const typename spec::activation_function aActivationFunction) :
            weighted_sum(spec::default_sum),
            activation_function(aActivationFunction)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = spec::default_weight;
        }

        /*!
            \brief Create a specified activation neuron with specified weighting of inputs
            \param aActivationFunction The function that controls the activation behaviour of this neuron
            \param aWeights The input weightings for this neuron
        */
        neuron(const typename spec::activation_function aActivationFunction, typename spec::const_weight_array& aWeights) :
            weights(aWeights),
            weighted_sum(spec::default_sum),
            activation_function(aActivationFunction)
        {}

        /*!
            \brief Create a specified activation neuron with specified weighting of all inputs
            \param aActivationFunction The function that controls the activation behaviour of this neuron
            \param aWeight The input weight for all inputs
        */
        neuron(const typename spec::activation_function aActivationFunction, const typename spec::weight_type aWeight) :
            weighted_sum(spec::default_sum),
            activation_function(aActivationFunction)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = spec::default_weight;
        }

        /*!
            \brief Create a specified behaviour neuron with equal weighting of all inputs
            \param aWeightedSum The function that controls the weighted sum behaviour of this neuron
            \param aActivationFunction The function that controls the activation behaviour of this neuron
        */
        neuron(const typename spec::weighted_sum aWeightedSum, const typename spec::activation_function aActivationFunction) :
            weighted_sum(aWeightedSum),
            activation_function(aActivationFunction)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = spec::default_weight;
        }

        /*!
            \brief Create a specified behaviour neuron with specified weighting of inputs
            \param aWeightedSum The function that controls the weighted sum behaviour of this neuron
            \param aActivationFunction The function that controls the activation behaviour of this neuron
            \param aWeights The input weightings for this neuron
        */
        neuron(const typename spec::weighted_sum aWeightedSum, const typename spec::activation_function aActivationFunction, typename spec::const_weight_array& aWeights) :
            weights(aWeights),
            weighted_sum(aWeightedSum),
            activation_function(aActivationFunction)
        {}

        /*!
            \brief Create a specified behaviour neuron with specified weighting of inputs
            \param aWeightedSum The function that controls the weighted sum behaviour of this neuron
            \param aActivationFunction The function that controls the activation behaviour of this neuron
            \param aWeight The input weight for all inputs
        */
        neuron(const typename spec::weighted_sum aWeightedSum, const typename spec::activation_function aActivationFunction, const typename spec::weight_type aWeight) :
            weighted_sum(aWeightedSum),
            activation_function(aActivationFunction)
        {
            for(int i = 0; i < spec::input_count; ++i) weights[i] = aWeight;
        }
    };

}}

#endif
