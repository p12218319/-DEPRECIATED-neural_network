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
#ifndef P12218319_ANN_LAYER_HPP
#define P12218319_ANN_LAYER_HPP

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	template<const uint32_t NEURON_COUNT_, class NEURON>
	class P12218319_EXPORT_API Layer : public LayerI {
	public:
		enum {
			NEURON_COUNT = NEURON_COUNT_
		};
		
		typedef NEURON NeuronType;
	private:
		NeuronType mNeurons[NEURON_COUNT];
	protected:
		// Inherited from LayerI
		NeuronI& GetNeuron(const uint32_t aIndex) throw() override {
			return mNeurons[aIndex];
		}
	public:
		// Inherited from LayerI
		
		uint32_t P12218319_CALL Size() const throw() override {
			return NEURON_COUNT;
		}
		
		void P12218319_CALL SetBias(const float aBias) throw() {
			for(uint32_t i = 0; i < NEURON_COUNT; ++i) mNeurons.SetBias(aBias);
		}
		
		void P12218319_CALL operator()(const float* const aInputs, float* const aOutputs) const throw() override {
			for(uint32_t i = 0; i < NEURON_COUNT; ++i) aOutputs[i] = mNeurons[i](aInputs);
		}
	};
}}

#endif