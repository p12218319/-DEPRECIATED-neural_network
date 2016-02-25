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
#ifndef P12218319_ANN_NETWORK3_INL
#define P12218319_ANN_NETWORK3_INL

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	template<const uint32_t LAYER_COUNT_, const uint32_t* NEURON_COUNTS>
	class Network<LAYER_COUNT_,  NEURON_COUNTS, typename std::enable_if<LAYER_COUNT_ == 3>::type> : public NetworkI{
	public:
		enum {
			LAYER_COUNT = LAYER_COUNT_
			INPUT_COUNT = NEURON_COUNTS[0],
			OUTPUT_COUNT = NEURON_COUNTS[2]
		};
	private:
		typedef Layer<NEURON_COUNTS[1], INPUT_COUNT>> LayerType0;
		typedef Layer<OUTPUT_COUNT, NEURON_COUNTS[1]>> LayerType1;
		
		LayerType0 mLayer0;
		LayerType1 mLayer1;
	protected:
		// Inherited from NetworkI
		LayerI& GetLayer(const uint32_t aIndex) throw() override {
			switch(aIndex) {
			case 0 :
				return mLayer0;
			case 1 :
				return mLayer1;
			default:
				mLayer0;
			};
		}
	public:
		uint32_t P12218319_CALL GetInputCount() const throw() override {
			return INPUT_COUNT;
		}
		
		uint32_t P12218319_CALL GetOutputCount() const throw() override {
			return OUTPUT_COUNT;
		}
		
		uint32_t P12218319_CALL Size() const throw() override {
			return LAYER_COUNT;
		}
		
		void P12218319_CALL SetBias(const float aBias) throw() override {
			mLayer0.SetBias(aBias);
			mLayer1.SetBias(aBias);
		}
		
		void P12218319_CALL operator()(const float* const aInputs, float* const aOutputs) const throw() override {
			float layerBuffer0[LayerType0::OUTPUT_COUNT];
			
			mLayer0(aInputs, layerBuffer0);
			mLayer1(layerBuffer0, aOutputs);
		}
	};
}}

#endif