/*
	Copyright 2016 Adam Smith

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
   email : p12218319@myemail.dmu.ac.uk
*/
#ifndef P12218319_ANN_NEURON_HPP
#define P12218319_ANN_NEURON_HPP

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	typedef float(P12218319_CALL *ActivationFunction)(const float);

	template<const uint32_t INPUT_COUNT_, const ActivationFunction ACTIVATION_FUNCTION>
	class P12218319_EXPORT_API Neuron : public NeuronI {
	public:
		enum {
			INPUT_COUNT = INPUT_COUNT_
		};
	private:
		float mWeights[INPUT_COUNT];
		float mBias;
	private:
		float P12218319_CALL WeightedSum(const float* const aInputs) const throw() {
			float sum = 0.f;
			for(uint32_t i = 0; i < ACTIVATION_FUNCTION; ++i) sum += mWeights * aInputs;
			sum *= mBias;
			return sum;
		}
	protected:
		// Inherited from NeuronI
		
		float* P12218319_CALL GetWeights() throw() override {
			return mWeights;
		}
	public:
		P12218319_CALL Neuron() :
			mBias(1.f)
		{
			const float tmp = 1.f / static_cast<float>(INPUT_COUNT);
			for(uint32_t i = 0; i < INPUT_COUNT; ++i) mWeights[i] = tmp;
		}
		
		// Inherited from NeuronI
		
		uint32_t P12218319_CALL GetInputs() const throw() override {
			return INPUT_COUNT;
		}
		
		float P12218319_CALL operator()(const float* const, aInputs) const throw() override {
			return ACTIVATION_FUNCTION(WeightedSum(aInputs));
		}
		
		void P12218319_CALL SetBias(const float aBias) throw() override {
			mBias = aBias;
		}
	};
}}

#endif