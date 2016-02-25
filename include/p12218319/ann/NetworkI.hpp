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
#ifndef P12218319_ANN_NETWORKI_HPP
#define P12218319_ANN_NETWORKI_HPP

#include "P12218319\core\Core.hpp"

namespace P12218319 { namespace ann {
	class P12218319_EXPORT_API NetworkI {
	protected:
		virtual LayerI& GetLayer(const uint32_t) throw() = 0;
	public:
		virtual P12218319_CALL ~NetworkI(){}
		
		virtual uint32_t P12218319_CALL GetInputCount() const throw() = 0;
		virtual uint32_t P12218319_CALL GetOutputCount() const throw() = 0;
		virtual uint32_t P12218319_CALL Size() const throw() = 0;
		virtual void P12218319_CALL SetBias(const float) throw() = 0;
		virtual void P12218319_CALL operator()(const float* const, float* const) const throw() = 0;
		
		inline LayerI& P12218319_CALL operator[](const uint32_t aIndex) throw() {return GetLayer(aIndex);}
		inline const LayerI& P12218319_CALL operator[](const uint32_t aIndex) const throw() {return const_cast<Layer*>(this)->GetLayer(aIndex);}
	};
}}

#endif