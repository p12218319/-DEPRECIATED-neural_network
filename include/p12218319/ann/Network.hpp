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
#ifndef P12218319_ANN_NETWORK_HPP
#define P12218319_ANN_NETWORK_HPP

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	template<const uint32_t LAYER_COUNT_, const uint32_t* NEURON_COUNTS, class ENABLE = void>
	class Network : public NetworkI {
	public:
		enum {
			LAYER_COUNT = LAYER_COUNT_
		};
	protected:
		// Inherited from NetworkI
		LayerI& GetLayer(const uint32_t) throw() override;
	public:
		uint32_t P12218319_CALL GetInputCount() const throw() override;
		uint32_t P12218319_CALL GetOutputCount() const throw() override;
		uint32_t P12218319_CALL Size() const throw() override;
		void P12218319_CALL SetBias(const float) throw() override;
		void P12218319_CALL operator()(const float* const, float* const) const throw() override;
	};
}}

#include "Network2.inl"
#include "Network3.inl"
#include "Network4.inl"

#endif