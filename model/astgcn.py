# -*- coding:utf-8 -*-
# pylint: disable=no-member

from mxnet import nd
from mxnet.gluon import nn


class Spatial_Attention_layer(nn.Block):
    '''
    compute spatial attention scores
    '''
    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.W_3 = self.params.get('W_3', allow_deferred_init=True)
            self.b_s = self.params.get('b_s', allow_deferred_init=True)
            self.V_s = self.params.get('V_s', allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)

        '''
        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer the shape of params
        self.W_1.shape = (num_of_timesteps, )
        self.W_2.shape = (num_of_features, num_of_timesteps)
        self.W_3.shape = (num_of_features, )
        self.b_s.shape = (1, num_of_vertices, num_of_vertices)
        self.V_s.shape = (num_of_vertices, num_of_vertices)
        for param in [self.W_1, self.W_2, self.W_3, self.b_s, self.V_s]:
            param._finish_deferred_init()

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = nd.dot(nd.dot(x, self.W_1.data()), self.W_2.data())

        # shape of rhs is (batch_size, T, V)
        rhs = nd.dot(self.W_3.data(), x.transpose((2, 0, 3, 1)))

        # shape of product is (batch_size, V, V)
        product = nd.batch_dot(lhs, rhs)

        S = nd.dot(self.V_s.data(),
                   nd.sigmoid(product + self.b_s.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normalization
        S = S - nd.max(S, axis=1, keepdims=True)
        exp = nd.exp(S)
        S_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return S_normalized


class cheb_conv_with_SAt(nn.Block):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''
    def __init__(self, num_of_filters, K, cheb_polynomials, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        self.Theta.shape = (self.K, num_of_features, self.num_of_filters)
        self.Theta._finish_deferred_init()

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = nd.zeros(shape=(batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):

                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta.data()[k]

                # shape is (batch_size, V, F)
                rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),
                                   graph_signal)

                output = output + nd.dot(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return nd.relu(nd.concat(*outputs, dim=-1))


class Temporal_Attention_layer(nn.Block):
    '''
    compute temporal attention scores
    '''
    def __init__(self, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.U_1 = self.params.get('U_1', allow_deferred_init=True)
            self.U_2 = self.params.get('U_2', allow_deferred_init=True)
            self.U_3 = self.params.get('U_3', allow_deferred_init=True)
            self.b_e = self.params.get('b_e', allow_deferred_init=True)
            self.V_e = self.params.get('V_e', allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer shape
        self.U_1.shape = (num_of_vertices, )
        self.U_2.shape = (num_of_features, num_of_vertices)
        self.U_3.shape = (num_of_features, )
        self.b_e.shape = (1, num_of_timesteps, num_of_timesteps)
        self.V_e.shape = (num_of_timesteps, num_of_timesteps)
        for param in [self.U_1, self.U_2, self.U_3, self.b_e, self.V_e]:
            param._finish_deferred_init()

        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = nd.dot(nd.dot(x.transpose((0, 3, 2, 1)), self.U_1.data()),
                     self.U_2.data())

        # shape is (N, V, T)
        rhs = nd.dot(self.U_3.data(), x.transpose((2, 0, 1, 3)))

        product = nd.batch_dot(lhs, rhs)

        E = nd.dot(self.V_e.data(),
                   nd.sigmoid(product + self.b_e.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normailzation
        E = E - nd.max(E, axis=1, keepdims=True)
        exp = nd.exp(E)
        E_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return E_normalized


class ASTGCN_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        '''
        Parameters
        ----------
        backbone: dict, should have 6 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",
                        "time_conv_strides",
                        "cheb_polynomials"
        '''
        super(ASTGCN_block, self).__init__(**kwargs)

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        with self.name_scope():
            self.SAt = Spatial_Attention_layer()
            self.cheb_conv_SAt = cheb_conv_with_SAt(
                num_of_filters=num_of_chev_filters,
                K=K,
                cheb_polynomials=cheb_polynomials)
            self.TAt = Temporal_Attention_layer()
            self.time_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                strides=(1, time_conv_strides))
            self.residual_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 1),
                strides=(1, time_conv_strides))
            self.ln = nn.LayerNorm(axis=2)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        # shape is (batch_size, T, T)
        temporal_At = self.TAt(x)

        x_TAt = nd.batch_dot(x.reshape(batch_size, -1, num_of_timesteps),
                             temporal_At)\
                  .reshape(batch_size, num_of_vertices,
                           num_of_features, num_of_timesteps)

        # cheb gcn with spatial attention
        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # convolution along time axis
        time_conv_output = (self.time_conv(spatial_gcn.transpose((0, 2, 1, 3)))
                            .transpose((0, 2, 1, 3)))

        # residual shortcut
        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))

        return self.ln(nd.relu(x_residual + time_conv_output))


class ASTGCN_submodule(nn.Block):
    '''
    a module in ASTGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(ASTGCN_submodule, self).__init__(**kwargs)

        self.blocks = nn.Sequential()
        for backbone in backbones:
            self.blocks.add(ASTGCN_block(backbone))

        with self.name_scope():
            # use convolution to generate the prediction
            # instead of using the fully connected layer
            self.final_conv = nn.Conv2D(
                channels=num_for_prediction,
                kernel_size=(1, backbones[-1]['num_of_time_filters']))
            self.W = self.params.get("W", allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray,
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, num_for_prediction)

        '''
        x = self.blocks(x)
        module_output = (self.final_conv(x.transpose((0, 3, 1, 2)))
                         [:, :, :, -1].transpose((0, 2, 1)))
        _, num_of_vertices, num_for_prediction = module_output.shape
        self.W.shape = (num_of_vertices, num_for_prediction)
        self.W._finish_deferred_init()
        return module_output * self.W.data()


class ASTGCN(nn.Block):
    '''
    ASTGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules
        '''
        super(ASTGCN, self).__init__(**kwargs)
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.submodules = []
        with self.name_scope():
            for backbones in all_backbones:
                self.submodules.append(
                    ASTGCN_submodule(num_for_prediction, backbones))
                self.register_child(self.submodules[-1])

    def forward(self, x_list):
        '''
        Parameters
        ----------
        x_list: list[mx.ndarray],
                shape is (batch_size, num_of_vertices,
                          num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: mx.ndarray,
               shape is (batch_size, num_of_vertices, num_for_prediction)

        '''
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return nd.add_n(*submodule_outputs)
