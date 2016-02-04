#ifndef P12218319_NEURAL_NETWORK_NEURON_WRAPPER_HPP
#define P12218319_NEURAL_NETWORK_NEURON_WRAPPER_HPP

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

#include <cstdint>
#include "neural_spec.hpp"
#include "neuron.hpp"

namespace p12218319 { namespace neural_network {

    namespace implementation {

		/*!
			\brief A function that calls neuron::operator()
			\detail This function is used internally by neuron_wrapper to implement it's operator()
			\tparam INPUT_TYPE The type of the neuron's inputs and outputs
			\tparam WEIGHT_TYPE The type of the neuron's weighting values
			\tparam INPUTS The number of inputs for the neuron
			\param aNeuron The neuron's address
			\param aInputs The inputs to pass to the neuron
			\return The output of the neuron
			\see neuron
			\see neuron_wrapper
		*/
        template<class INPUT_TYPE, class WEIGHT_TYPE, const int INPUTS>
        INPUT_TYPE neuron_wrapper_run(const void* const aNeuron, const INPUT_TYPE* const aInputs) throw() {
            typedef const INPUT_TYPE input_array[INPUTS];
            return static_cast<const neuron<neural_spec<INPUT_TYPE, WEIGHT_TYPE, INPUTS>>*>(aNeuron)->operator()(reinterpret_cast<input_array&>(aInputs));
        }

		/*!
			\brief A function that calls neuron::weights::operator[]
			\detail This function is used internally by neuron_wrapper::weight_array to implement it's operator[]
			\tparam INPUT_TYPE The type of the neuron's inputs and outputs
			\tparam WEIGHT_TYPE The type of the neuron's weighting values
			\tparam INPUTS The number of inputs for the neuron
			\param aNeuron The neuron's address
			\param aIndex The index of the weight to get
			\return A reference to the weighting value
			\see neuron
			\see neuron_wrapper
		*/
        template<class INPUT_TYPE, class WEIGHT_TYPE, const int INPUTS>
        WEIGHT_TYPE& neuron_wrapper_get_weight(void* const aNeuron, const int aIndex) throw() {
            return static_cast<neuron<neural_spec<INPUT_TYPE, WEIGHT_TYPE, INPUTS>>*>(aNeuron)->weights[aIndex];
        }
    }

	/*!
		\class neuron_wrapper
		\brief A wrapper class that removes the template requirements of a neuron object, allowing for runtime examination of a neuron
		\detail The input count for the neuron is determined automatically by the constructor.
		\tparam INPUT_TYPE The type of the neuron's inputs and outputs
		\tparam WEIGHT_TYPE The type of the neuron's weighting values
		\version 1.0
		\see neuron
	*/
    template<class INPUT_TYPE, class WEIGHT_TYPE>
    class neuron_wrapper {
    private:
        void* mNeuron;																//!< The address of the referenced neuron object
        INPUT_TYPE (*mRun)(const void* const, const INPUT_TYPE* const aInputs);		//!< A function that calls neuron::operator() on mNeuron
        WEIGHT_TYPE& (*mGetWeight)(void* const, const int);							//!< A function that calls neuron::weights::operator[] on mNeuron
        uint8_t mWeights;															//!< Contains the number of weights mNeuron has
    public:
		/*!
			\class weight_array
			\brief A class that provides the same interface for accessing weighting values as neuron::weights
			\version 1.0
		*/
        class weight_array{
        private:
            neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>& mNeuron;	//!< The neuron_wrapper that owns this array
        public:

            // Functions

			/*!
				\brief Get the number of inputs for the neuron
				\return The input count
			*/
            inline int size() const throw() {
                return mNeuron.mWeights;
            }

            // Operators

			/*!
				\brief Access a weighting value of the parent neuron
				\param aIndex The index of the weight to get
				\return A reference to the weighting value
			*/
            inline WEIGHT_TYPE& operator[](const int aIndex) throw() {
                return mNeuron.mGetWeight(mNeuron.mNeuron, aIndex);
            }

			/*!
				\brief Access a weighting value of the parent neuron
				\param aIndex The index of the weight to get
				\return The weighting value
			*/
            inline const WEIGHT_TYPE& operator[](const int aIndex) const throw() {
                return mNeuron.mGetWeight(const_cast<void*>(mNeuron.mNeuron), aIndex);
            }

            // Constructors

			/*!
				\brief Create a new weight_array
				\param aNeuron The parent neuron_wrapper that owns this array
			*/
            weight_array(neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>& aNeuron) throw() :
                mNeuron(aNeuron)
            {}
        };

        weight_array weights;	//!< Wrapper class for accessing weighting values with the same semantics as neuron
    public:

        // Operators

		/*!
			\brief Calls neuron::operator()
			\param aInputs The inputs to the neuron
			\return The output of the neuron
			\see neuron::operator()
		*/
        inline INPUT_TYPE operator()(const INPUT_TYPE* const aInputs) const throw() {
            return mRun(mNeuron, aInputs);
        }

        // Constructors

		/*!
			\brief Create a neuron_wrapper that references an existing neuron object
			\tparam INPUTS The number of inputs for /a aNeuron, this will be determined automatically by the compiler
			\param aNeuron A reference to the neuron to defer calls to
		*/
        template<const int INPUTS>
        neuron_wrapper(neuron<neural_spec<INPUT_TYPE, WEIGHT_TYPE, INPUTS>>& aNeuron) throw() :
            mNeuron(&aNeuron),
            mRun(&implementation::neuron_wrapper_run<INPUT_TYPE, WEIGHT_TYPE, INPUTS>),
            mGetWeight(&implementation::neuron_wrapper_get_weight<INPUT_TYPE, WEIGHT_TYPE, INPUTS>),
            mWeights(INPUTS),
            weights(*this)
        {}

		/*!
			\brief Neuron copy constructor
			\param aOther The neuron to copy
		*/
        neuron_wrapper(const neuron_wrapper<INPUT_TYPE, WEIGHT_TYPE>& aOther) throw() :
            mNeuron(aOther.mNeuron),
            mRun(aOther.mRun),
            mGetWeight(aOther.mGetWeight),
            mWeights(aOther.mWeights),
            weights(*this)
        {}
    };

}}

#endif
