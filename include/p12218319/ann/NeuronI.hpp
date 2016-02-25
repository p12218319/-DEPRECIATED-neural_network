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
#ifndef P12218319_ANN_NEURONI_HPP
#define P12218319_ANN_NEURONI_HPP

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	class P12218319_EXPORT_API NeuronI {
	protected:
		virtual float* P12218319_CALL GetWeights() throw() = 0;
	public:
		virtual P12218319_CALL ~NeuronI(){}
		
		virtual uint32_t P12218319_CALL GetInputs() const throw() = 0;
		virtual float P12218319_CALL operator()(const float* const) const throw() = 0;
		virtual void P12218319_CALL SetBias(const float) throw() = 0;
		
		inline float& P12218319_CALL operator[](const uint32_t aWeight) throw() {return GetWeights()[aWeight];}
		inline float P12218319_CALL operator[](const uint32_t aWeight) const throw() {return const_cast<NeuronI*>(this)->GetWeights()[aWeight];}
		inline void P12218319_CALL SetBias(const float aBias) throw() {const uint32_t count = Size(); for(uint32_t i = 0; i < count; ++i) GetNeuron(i).SetBias(aBias);}
	};
}}

#endif