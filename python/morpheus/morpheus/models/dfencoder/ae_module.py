# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict

import torch


def _compute_embedding_size(n_categories):
    """Applies a standard formula to choose the number of feature embeddings
    to use in a given embedding layers.

    Parameters
    ----------
    n_categories : int
        number of unique categories in a column

    Returns
    -------
    int
        the coputed embedding size
    """
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)


class CompleteLayer(torch.nn.Module):
    """Impliments a layer with linear transformation and optional activation and dropout. """

    def __init__(self, in_dim, out_dim, activation=None, dropout=None, *args, **kwargs):
        """ 
        Initializes a CompleteLayer object with given input and output dimensions, activation function, and dropout probability.

        Parameters
        ----------
        in_dim : int
            The size of the input dimension.
        out_dim : int
            The size of the output dimension.
        activation : str, optional
            The name of the activation function to use. Defaults to None if no activation function is desired.
        dropout : float, optional
            The probability of dropout to apply. Defaults to None if no dropout is desired.
        *args : list
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.layers = []
        linear = torch.nn.Linear(in_dim, out_dim)
        self.layers.append(linear)
        self.add_module("linear_layer", linear)

        self.activation = activation
        if activation is not None:
            act = self.interpret_activation(activation)
            self.layers.append(act)
        if dropout is not None:
            dropout_layer = torch.nn.Dropout(dropout)
            self.layers.append(dropout_layer)
            self.add_module("dropout", dropout_layer)

    def interpret_activation(self, act=None):
        """
        Interprets the name of the activation function and returns the appropriate PyTorch function.

        Parameters
        ----------
        act : str, optional
            The name of the activation function to interpret. Defaults to None if no activation function is desired.

        Raises
        ------
        Exception
            If the activation function name is not recognized.

        Returns
        -------
        PyTorch function
            The PyTorch activation function that corresponds to the given name.
        """
        if act is None:
            act = self.activation
        activations = {
            "leaky_relu": torch.nn.functional.leaky_relu,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "selu": torch.selu,
            "hardtanh": torch.nn.functional.hardtanh,
            "relu6": torch.nn.functional.relu6,
            "elu": torch.nn.functional.elu,
            "celu": torch.nn.functional.celu,
            "rrelu": torch.nn.functional.rrelu,
            "hardshrink": torch.nn.functional.hardshrink,
            "tanhshrink": torch.nn.functional.tanhshrink,
            "softsign": torch.nn.functional.softsign,
        }
        try:
            return activations[act]
        except:
            msg = f"activation {act} not understood. \n"
            msg += "please use one of: \n"
            msg += str(list(activations.keys()))
            raise Exception(msg)

    def forward(self, x):
        """Performs a forward pass through the CompleteLayer object.

        Parameters
        ----------
        x : tensor
            The input tensor to the CompleteLayer object.

        Returns
        -------
        torch.Tensor
            The output tensor of the CompleteLayer object after processing the input through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class AEModule(torch.nn.Module):
    """ Auto Encoder Pytorch Module."""

    def __init__(
        self,
        verbose,
        encoder_layers=None,
        decoder_layers=None,
        encoder_dropout=None,
        decoder_dropout=None,
        encoder_activations=None,
        decoder_activations=None,
        activation="relu",
        device=None,
        *args,
        **kwargs,
    ):
        """Initializes an instance of the `AEModule` class.

        Parameters
        ----------
        verbose : bool
            If True, log information during the construction of the model.
        encoder_layers : list[int], optional
            List of hidden layer sizes for the encoder. If not given, a default-sized encoder is used. Defaults to None.
        decoder_layers : list[int], optional
            List of hidden layer sizes for the decoder. Defaults to None.
        encoder_dropout : Union[float, List[float]], optional
            The dropout rate(s) for the encoder layer(s). Defaults to None.
        decoder_dropout : Union[float, List[float]], optional
            The dropout rate(s) for the decoder layer(s). Defaults to None.
        encoder_activations : List[str], optional
            The activation function(s) for the encoder layer(s). Defaults to None.
        decoder_activations : List[str], optional
            The activation function(s) for the decoder layer(s). Defaults to None.
        activation : str, optional
            The default activation function used for encoder and decoder layers if not specified in
            encoder_activations or decoder_activations. Defaults to "relu".
        device : str, optional
            The device to run the model on.
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.activation = activation
        self.device = device

        # mapping cat features to the embedding layer
        self.categorical_embedding = OrderedDict()
        self.encoder = []
        self.decoder = []
        self.numeric_output = None
        self.binary_output = None
        # mapping cat features to the output layer
        self.categorical_output = OrderedDict()

    def build(self, numeric_fts, binary_fts, categorical_fts):
        """Constructs the autoencoder model.

        Parameters
        ----------
        numeric_fts : List[str]
            The names of the numeric features.
        binary_fts : List[str]
            The names of the binary features.
        categorical_fts : Dict[str, Dict[str, List[str]]]
            The dictionary mapping categorical feature names to dictionaries containing the categories of the feature.
        """
        if self.verbose:
            logging.info("Building model...")

        cat_input_dim = self._build_categorical_input_layers(categorical_fts)

        # compute input dimension
        num_ft_cnt, bin_ft_cnt = len(numeric_fts), len(binary_fts)
        input_dim = cat_input_dim + num_ft_cnt + bin_ft_cnt

        dim = self._build_layers(input_dim)

        # set up predictive outputs
        self._build_outputs(dim, num_ft_cnt, bin_ft_cnt, categorical_fts)

        self.to(self.device)

    def _build_categorical_input_layers(self, categorical_fts):
        """Builds the categorical input layers of the autoencoder model.

        Parameters
        ----------
        categorical_fts : dict
            The dictionary mapping categorical feature names to dictionaries containing the categories of the feature.
            The second-layer dictionaries have a key "cats" which maps to a list containing the actual categorical values.

        Returns
        -------
        int
            The total dimensions of the categorical features combined.
        """
        # will compute total number of inputs
        input_dim = 0

        # create categorical variable embedding layers
        for ft, feature in categorical_fts.items():
            n_cats = len(feature["cats"]) + 1
            embed_dim = _compute_embedding_size(n_cats)
            embed_layer = torch.nn.Embedding(n_cats, embed_dim)
            self.categorical_embedding[ft] = embed_layer
            self.add_module(f"{ft}_embedding", embed_layer)
            # track embedding inputs
            input_dim += embed_dim

        return input_dim

    def _build_layers(self, input_dim):
        """Constructs the encoder and decoder layers for the autoencoder model.

        Parameters
        ----------
        input_dim : int
            The input dimension of the autoencoder model.

        Returns
        -------
        int
            The output dimension of the encoder layers.
        """
        # construct a canned denoising autoencoder architecture
        if self.encoder_layers is None:
            self.encoder_layers = [int(4 * input_dim) for _ in range(3)]

        if self.decoder_layers is None:
            self.decoder_layers = []

        if self.encoder_activations is None:
            self.encoder_activations = [self.activation for _ in self.encoder_layers]

        if self.decoder_activations is None:
            self.decoder_activations = [self.activation for _ in self.decoder_layers]

        if self.encoder_dropout is None or type(self.encoder_dropout) == float:
            drp = self.encoder_dropout
            self.encoder_dropout = [drp for _ in self.encoder_layers]

        if self.decoder_dropout is None or type(self.decoder_dropout) == float:
            drp = self.decoder_dropout
            self.decoder_dropout = [drp for _ in self.decoder_layers]

        for i, dim in enumerate(self.encoder_layers):
            activation = self.encoder_activations[i]
            layer = CompleteLayer(input_dim, dim, activation=activation, dropout=self.encoder_dropout[i])
            input_dim = dim
            self.encoder.append(layer)
            self.add_module(f"encoder_{i}", layer)

        for i, dim in enumerate(self.decoder_layers):
            activation = self.decoder_activations[i]
            layer = CompleteLayer(input_dim, dim, activation=activation, dropout=self.decoder_dropout[i])
            input_dim = dim
            self.decoder.append(layer)
            self.add_module(f"decoder_{i}", layer)

        return input_dim

    def _build_outputs(self, dim, num_ft_cnt, bin_ft_cnt, categorical_fts):
        """Construct the output of the model from its inputs.

        Parameters
        ----------
        dim : int
            The dimensionality of the input features.
        num_ft_cnt : int
            The number of numeric features in the input.
        bin_ft_cnt : int
            The number of binary features in the input.
        categorical_fts : Dict[str, Dict[str, List[str]]]
            The dictionary mapping categorical feature names to dictionaries containing the categories of the feature. 
            The second-layer dictionaries have a key "cats" which maps to a list containing the actual categorical values.
        """
        self.numeric_output = torch.nn.Linear(dim, num_ft_cnt)
        self.binary_output = torch.nn.Linear(dim, bin_ft_cnt)

        for ft, feature in categorical_fts.items():
            cats = feature["cats"]
            layer = torch.nn.Linear(dim, len(cats) + 1)
            self.categorical_output[ft] = layer
            self.add_module(f"{ft}_output", layer)

    def forward(self, input):
        """Passes the input through the model and returns the outputs.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple of Union[torch.Tensor, List[torch.Tensor]]
            A tuple containing the numeric (Tensor), binary (Tensor), and categorical outputs (List[torch.Tensor]) of the model.
        """
        encoding = self.encode(input)
        num, bin, cat = self.decode(encoding)
        return num, bin, cat

    def encode(self, x, layers=None):
        """Encodes the input using the encoder layers.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to encode.
        layers : int, optional
            The number of layers to use for encoding. Defaults to None, will use all encoder layers.

        Returns
        -------
        torch.Tensor
            The encoded output tensor.
        """
        if layers is None:
            layers = len(self.encoder)
        for i in range(layers):
            layer = self.encoder[i]
            x = layer(x)
        return x

    def decode(self, x, layers=None):
        """Decodes the input using the decoder layers and computes the outputs.

        Parameters
        ----------
        x : torch.Tensor
            The encoded input tensor to decode.
        layers : int, optional
            The number of layers to use for decoding. Defaults to None, will use all decoder layers.

        Returns
        -------
        tuple of Union[torch.Tensor, List[torch.Tensor]]
            A tuple containing the numeric (Tensor), binary (Tensor), and categorical outputs (List[torch.Tensor]) of the model.
        """
        if layers is None:
            layers = len(self.decoder)
        for i in range(layers):
            layer = self.decoder[i]
            x = layer(x)
        num, bin, cat = self._compute_outputs(x)
        return num, bin, cat

    def _compute_outputs(self, x):
        """Computes the numeric, binary, and categorical outputs from the decoded input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The decoded input tensor.

        Returns
        -------
        tuple of Union[torch.Tensor, List[torch.Tensor]]
            A tuple containing the numeric (Tensor), binary (Tensor), and categorical outputs (List[torch.Tensor]) of the model.
        """
        num = self.numeric_output(x)
        bin = self.binary_output(x)
        bin = torch.sigmoid(bin)
        cat = []
        for output_layer in self.categorical_output.values():
            out = output_layer(x)
            cat.append(out)
        return num, bin, cat
