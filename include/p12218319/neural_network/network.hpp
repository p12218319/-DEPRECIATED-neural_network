#ifndef P12218319_NEURAL_NETWORK_NETWORK_HPP
#define P12218319_NEURAL_NETWORK_NETWORK_HPP

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

#include "layer_wrapper.hpp"
#include "layer.hpp"

namespace p12218319 { namespace neural_network {

    namespace implementation {
	
		/*!
			\brief Helper class for defining layer types
			\tparam LAYER The layer of the network to define
			\tparam INPUT_TYPE The input and output types for the neurons in this network
			\tparam WEIGHT_TYPE The weighting value type for the neurons in this network
			\tparam LAYERS The number of layers in this network, excluding the output layer
			\tparam LAYER_NEURONS An array containing the number of outputs for each layer, including the output layer
			\tparam ENABLE Used to automatically select the correct specialisation of this class
			\version 1.0
			\see network
		
		*/
        template<const int LAYER, class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1], typename ENABLE = void>
        struct network_layer{
            typedef layer<neural_spec<INPUT_TYPE, WEIGHT_TYPE, LAYER_NEURONS[LAYER - 1]> , LAYER_NEURONS[LAYER]> type;	//!< A layer type with correctly defined types and input / output counts
        };


        template<const int LAYER, class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1]>
        struct network_layer<LAYER, INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS, typename std::enable_if<LAYER == 0>::type>{
            typedef layer<neural_spec<INPUT_TYPE, WEIGHT_TYPE, LAYER_NEURONS[LAYER]> , LAYER_NEURONS[LAYER]> type;
        };


        template<const int LAYER, class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1]>
        struct network_layer<LAYER, INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS, typename std::enable_if<LAYER == LAYERS>::type>{
            typedef layer<neural_spec<INPUT_TYPE, WEIGHT_TYPE, LAYER_NEURONS[LAYER - 1]> , LAYER_NEURONS[LAYER]> type;
        };
    }

	/*!
		\brief A feed-forward neural network
		\tparam INPUT_TYPE The input and output types for the neurons in this network
		\tparam WEIGHT_TYPE The weighting value type for the neurons in this network
		\tparam LAYERS The number of layers in this network, excluding the output layer
		\tparam LAYER_NEURONS An array containing the number of outputs for each layer, including the output layer
		\tparam ENABLE Used to automatically select the correct specialisation of this class
		\version 1.0
		\see layer
	*/
    template<class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1],  typename ENABLE = void>
    class network {
	public :
		/*!
			\brief The layer class of a specified layer
			\tparam LAYER The index of the layer to define
			\see layer
		*/
        template<const int LAYER>
        using layer_type = void;
		
        enum {
            input_count = 0,	//!< The number of inputs to this network
            output_count = 0	//!< The number of outputs from this network
        };

        typedef INPUT_TYPE input_array[input_count];							//!< An array that can contain the inputs to this network
        typedef const INPUT_TYPE const_input_array[input_count];				//!< A read-only array that can contain the inputs to this network
        typedef INPUT_TYPE output_array[output_count];							//!< An array that can contain the outputs of this network
        typedef const INPUT_TYPE const_output_array[output_count];				//!< A read-only array that can contain the outputs of this network
        typedef network<INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS> self;	//!< A shortcut that refers to this network type
		
        // Functions

		/*!
			\brief Get a reference to a layer of the network
			\tparam LAYER The index of the layer to get
			\return The layer
		*/
        template<const int LAYER>
        inline layer_type<LAYER>& get_layer() throw();

		/*!
			\brief Get a reference to a layer of the network
			\tparam LAYER The index of the layer to get
			\return The layer
		*/
        template<const int LAYER>
        inline const layer_type<LAYER>& get_layer() const throw();
		

		/*!
			\brief Get a reference to a layer of the network
			\param aLayer The index of the layer to get
			\return The layer wrapper
			\see layer_wrapper
			\see generic_node
		*/
        layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) throw();

		/*!
			\brief Get a reference to a layer of the network
			\param aLayer The index of the layer to get
			\return The layer wrapper
			\see layer_wrapper
			\see generic_node
		*/
        const layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) const throw();

        // Operators

		/*!
			\brief Call each layer in the network
			\param aInputs Contains the inputs to the networks
			\param aOutputs Contains the outputs of the network
		*/
        void operator()(const_input_array& aInputs, output_array& aOutputs);}

	};
	
    template<class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1]>
    class network<INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS, typename std::enable_if<LAYERS == 1>::type> {
    public:
        template<const int LAYER>
        using layer_type = typename implementation::network_layer<LAYER, INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS>::type;

        enum {
            input_count = layer_type<0>::spec::input_count,
            output_count = layer_type<0>::output_count
        };

        typedef INPUT_TYPE input_array[input_count];
        typedef const INPUT_TYPE const_input_array[input_count];
        typedef INPUT_TYPE output_array[output_count];
        typedef const INPUT_TYPE const_output_array[output_count];
        typedef network<INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS> self;
    private:
        layer_type<0> mInputLayer;
        layer_type<1> mOutputLayer;
    public:

        // Functions

        template<const int LAYER>
        inline typename std::enable_if<LAYER == 0, layer_type<LAYER>&>::type get_layer() throw() {
            return mInputLayer;
        }

        template<const int LAYER>
        inline typename std::enable_if<LAYER == 1, layer_type<LAYER>&>::type get_layer() throw() {
            return mOutputLayer;
        }

        template<const int LAYER>
        inline const layer_type<LAYER>& get_layer() const throw() {
            return const_cast<self*>(this)->get_layer<LAYER>();
        }

        layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) throw() {
            switch(aLayer) {
            case 0:
                return layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>(mInputLayer);
            default:
                return layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>(mOutputLayer);
            }
        }

        const layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) const throw() {
            return const_cast<self*>(this)->get_layer(aLayer);
        }

        // Operators

        void operator()(const_input_array& aInputs, output_array& aOutputs) {
            typename layer_type<0>::output_array out0;
            mInputLayer(aInputs, out0);
            mOutputLayer(out0, aOutputs);
        }
    };

    template<class INPUT_TYPE, class WEIGHT_TYPE, const int LAYERS, const int LAYER_NEURONS[LAYERS + 1]>
    class network<INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS, typename std::enable_if<LAYERS == 2>::type> {
    public:
        template<const int LAYER>
        using layer_type = typename implementation::network_layer<LAYER, INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS>::type;

        enum {
            input_count = layer_type<0>::spec::input_count,
            output_count = layer_type<2>::output_count
        };

        typedef INPUT_TYPE input_array[input_count];
        typedef const INPUT_TYPE const_input_array[input_count];
        typedef INPUT_TYPE output_array[output_count];
        typedef const INPUT_TYPE const_output_array[output_count];
        typedef network<INPUT_TYPE, WEIGHT_TYPE, LAYERS, LAYER_NEURONS> self;
    private:
        layer_type<0> mInputLayer;
        layer_type<1> mHiddenLayer0;
        layer_type<2> mOutputLayer;
    public:

        // Functions

        template<const int LAYER>
        inline typename std::enable_if<LAYER == 0, layer_type<LAYER>&>::type get_layer() throw() {
            return mInputLayer;
        }

        template<const int LAYER>
        inline typename std::enable_if<LAYER == 1, layer_type<LAYER>&>::type get_layer() throw() {
            return mHiddenLayer0;
        }

        template<const int LAYER>
        inline typename std::enable_if<LAYER == 2, layer_type<LAYER>&>::type get_layer() throw() {
            return mOutputLayer;
        }

        template<const int LAYER>
        inline const layer_type<LAYER>& get_layer() const throw() {
            return const_cast<self*>(this)->get_layer<LAYER>();
        }

        // Operators

        void operator()(const_input_array& aInputs, output_array& aOutputs) {
            typename layer_type<0>::output_array out0;
            mInputLayer(aInputs, out0);
            typename layer_type<1>::output_array out1;
            mHiddenLayer0(out0, out1);
            mOutputLayer(out1, aOutputs);
        }

        layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) throw() {
            switch(aLayer) {
            case 0:
                return layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>(mInputLayer);
            case 1:
                return layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>(mHiddenLayer0);
            default:
                return layer_wrapper<INPUT_TYPE, WEIGHT_TYPE>(mOutputLayer);
            }
        }

        const layer_wrapper<INPUT_TYPE, WEIGHT_TYPE> get_layer(const int aLayer) const throw() {
            return const_cast<self*>(this)->get_layer(aLayer);
        }
    };

}}

#endif
