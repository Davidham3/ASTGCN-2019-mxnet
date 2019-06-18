# -*- coding:utf-8 -*-
# pylint: disable=no-member

from mxnet import nd
from mxnet.gluon import nn


class cheb_conv(nn.Block):
    '''
    K-order chebyshev graph convolution
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
        super(cheb_conv, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix,
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

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
            graph_signal = x[:, :, :, time_step]

            output = nd.zeros(shape=(batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                theta_k = self.Theta.data()[k]
                rhs = nd.dot(graph_signal.transpose((0, 2, 1)),
                             T_k).transpose((0, 2, 1))
                output = output + nd.dot(rhs, theta_k)
            outputs.append(output.expand_dims(-1))

        return nd.relu(nd.concat(*outputs, dim=-1))


class MSTGCN_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        '''
        Parameters
        ----------
        backbone: dict, should have 5 keys
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_strides",
                        "cheb_polynomials"
        '''
        super(MSTGCN_block, self).__init__(**kwargs)

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        with self.name_scope():
            self.cheb_conv = cheb_conv(num_of_filters=num_of_chev_filters,
                                       K=K,
                                       cheb_polynomials=cheb_polynomials)
            self.time_conv = nn.Conv2D(channels=num_of_time_filters,
                                       kernel_size=(1, 3),
                                       padding=(0, 1),
                                       strides=(1, time_conv_strides))
            self.residual_conv = nn.Conv2D(channels=num_of_time_filters,
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

        # cheb gcn
        spatial_gcn = self.cheb_conv(x)

        # convolution along time axis
        time_conv_output = (self.time_conv(spatial_gcn.transpose((0, 2, 1, 3))
                            .transpose((0, 2, 1, 3))))

        # residual shortcut
        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))

        return self.ln(nd.relu(x_residual + time_conv_output))


class MSTGCN_submodule(nn.Block):
    '''
    a module in MSTGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(MSTGCN_submodule, self).__init__(**kwargs)

        self.blocks = []
        for backbone in backbones:
            self.blocks.append(MSTGCN_block(backbone))
            self.register_child(self.blocks[-1])

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
        x: mx.ndarray
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, num_for_prediction)

        '''

        for block in self.blocks:
            x = block(x)
        module_output = (self.final_conv(x.transpose((0, 3, 1, 2)))
                         [:, :, :, -1].transpose((0, 2, 1)))
        _, num_of_vertices, num_for_prediction = module_output.shape
        self.W.shape = (num_of_vertices, num_for_prediction)
        self.W._finish_deferred_init()
        return module_output * self.W.data()


class MSTGCN(nn.Block):
    '''
    MSTGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules
        '''
        super(MSTGCN, self).__init__(**kwargs)
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.num_for_prediction = num_for_prediction

        self.submodules = []
        with self.name_scope():
            for backbones in all_backbones:
                self.submodules.append(
                    MSTGCN_submodule(num_for_prediction, backbones))
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
                             "Check if your input data have same "
                             "size on axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return nd.add_n(*submodule_outputs)
