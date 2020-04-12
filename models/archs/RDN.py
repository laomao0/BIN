import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init as weight_init
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell

    Notice the input_size, hidden_size is the channel size.

    """
    def __init__(self, input_size, hidden_size,  forget_bias=1.0, kernel_size=3, padding=3//2):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding= padding,bias=True)

        self._forget_bias = forget_bias
        self._initialize_weights()

    def _initialize_weights(self):  # The Xavier method of initialize
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(m)
                count +=1
                # print(count)
                weight_init.xavier_uniform_(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)
    # def zero_state(self,batch_size,  h, w, ):
    #     zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
    #     return zeros
    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if input_.is_cuda:
                prev_state = [
                    torch.cuda.FloatTensor().resize_(state_size).zero_(),
                    torch.cuda.FloatTensor().resize_(state_size).zero_(),
                    ]
            else:
                prev_state = [
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                ]

        # prev_hidden, prev_cell = prev_state
        c, h = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, h), 1)
        concat = self.Gates(stacked_inputs)

        # chunk across channel dimension
        # in_gate,cell_gate, remember_gate, out_gate= torch.split(concat, 4, 1) #gates.chunk(4, 1)
        i, j,f, o = concat.chunk(4,1) # torch.split(concat, 4, 1) #gates.chunk(4, 1)

        new_c = (c * torch.sigmoid(f + self._forget_bias) + torch.sigmoid(i).mul(torch.tanh(j)))
        new_h = torch.tanh(new_c).mul(torch.sigmoid(o))
        # apply sigmoid non linearity
        # in_gate = torch.sigmoid(in_gate)
        # remember_gate = torch.sigmoid(remember_gate)
        # out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        # cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        # cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        # hidden = out_gate * torch.tanh(cell)

        return new_h, [new_c, new_h]

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))
def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN_residual_interp_2_input(nn.Module):
    def __init__(self,
                 G0=64,
                 D=6,
                 C=4,
                 G=32):
        super(RDN_residual_interp_2_input, self).__init__()
        # self.G0 = 64
        self.G0 = G0
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        # self.D = 6
        self.D = D
        # self.C = 4
        self.C = C
        # self.G = 32
        self.G = G

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(24, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate = self.G, nConvLayers=self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, B0, B1):
        B_shuffle = pixel_reshuffle(torch.cat((B0, B1), 1), 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        F = self.UPNet(x) + (B0 + B1) / 2
        return F

class RDN_residual_interp_2_1_input(nn.Module):
    def __init__(self,
                 G0=64,
                 D=6,
                 C=4,
                 G=32
                 ):
        super(RDN_residual_interp_2_1_input, self).__init__()
        # self.G0 = 64
        self.G0 = G0
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        # self.D = 6
        # self.C = 4
        # self.G = 32
        self.D = D
        self.C = C
        self.G = G

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(36, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = self.G0, growRate = self.G, nConvLayers = self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, I0, I1, I2):
        B_shuffle = pixel_reshuffle(torch.cat((I0, I1, I2), 1), 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        F = self.UPNet(x) + (I0 + I1 + I2) / 3
        return F

class RDN_residual_interp_4_1_input(nn.Module):
    def __init__(self,
                 G0=64,
                 D=6,
                 C=4,
                 G=32
                 ):
        super(RDN_residual_interp_4_1_input, self).__init__()
        self.G0 = G0
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D = D
        self.C = C
        self.G = G

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(60, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = self.G0, growRate = self.G, nConvLayers = self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, B0, B1, B2, B3, B4):
        B_shuffle = pixel_reshuffle(torch.cat((B0, B1, B2, B3, B4), 1), 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        F = self.UPNet(x) + (B0 + B1 + B2 + B3 + B4) / 5
        return F


class RDN_residual_interp_5_input(nn.Module):
    def __init__(self,lstm=False, GO=64, D=6):
        super(RDN_residual_interp_5_input, self).__init__()
        # stage 1
        self.lstm = lstm
        self.model1_1 = RDN_residual_interp_2_input(G0=GO, D=D)
        self.model1_2 = self.model1_1
        self.model1_3 = self.model1_1
        self.model1_4 = self.model1_1
        # stage 2
        if lstm == True:
            self.model2_1 = RDN_residual_interp_2_1_input(G0=GO, D=D)  # three inputs
        else:
            self.model2_1 = RDN_residual_interp_2_input(G0=GO, D=D)
        self.model2_2 = self.model2_1
        self.model2_3 = self.model2_1
        # stage 3

        if lstm == True:
            self.model3_1 = RDN_residual_interp_4_1_input(G0=GO, D=D)
        else:
            self.model3_1 = RDN_residual_interp()
        self.model3_2 = self.model3_1
        # stage 4

        if lstm == True:
            self.model4_1 = RDN_residual_interp_4_1_input(G0=GO, D=D)
        else:
            self.model4_1 = RDN_residual_interp()

    def forward(self, B1, B3, B5, B7, B9, previous_input=None):

        if self.lstm == True:

            self.I2_prime = self.model1_1(B1, B3)
            self.I4_prime = self.model1_2(B3, B5)
            self.I6_prime = self.model1_3(B5, B7)
            self.I8_prime = self.model1_4(B7, B9)
            if previous_input[0] is not None:
                p4_prime, p6_prime, p8_prime, p5_prime_prime, p7_prime_prime, p6_prime_prime_prime = previous_input
                self.I3_prime = self.model2_1(p4_prime, self.I2_prime, self.I4_prime)
                self.I5_prime = self.model2_2(p6_prime, self.I4_prime, self.I6_prime)
                self.I7_prime = self.model2_3(p8_prime, self.I6_prime, self.I8_prime)
                self.I4_prime_prime = self.model3_1(p5_prime_prime, B3, self.I3_prime, self.I5_prime, B5)
                self.I6_prime_prime = self.model3_2(p7_prime_prime, B5, self.I5_prime, self.I7_prime, B7)
                self.I5_prime_prime = self.model4_1(p6_prime_prime_prime, self.I4_prime, self.I4_prime_prime, self.I6_prime_prime,self.I6_prime)
            else:
                self.I3_prime = self.model2_1(self.I2_prime, self.I2_prime, self.I4_prime)
                self.I5_prime = self.model2_2(self.I4_prime, self.I4_prime, self.I6_prime)
                self.I7_prime = self.model2_3(self.I6_prime, self.I6_prime, self.I8_prime)
                self.I4_prime_prime = self.model3_1(self.I3_prime, B3, self.I3_prime, self.I5_prime, B5)
                self.I6_prime_prime = self.model3_2(self.I5_prime, B5, self.I5_prime, self.I7_prime, B7)
                self.I5_prime_prime = self.model4_1(self.I4_prime, self.I4_prime, self.I4_prime_prime,self.I6_prime_prime, self.I6_prime)

        else:
            self.I2_prime = self.model1_1(B1, B3)
            self.I4_prime = self.model1_2(B3, B5)
            self.I6_prime = self.model1_3(B5, B7)
            self.I8_prime = self.model1_4(B7, B9)
            self.I3_prime = self.model2_1(self.I2_prime, self.I4_prime)
            self.I5_prime = self.model2_2(self.I4_prime, self.I6_prime)
            self.I7_prime = self.model2_3(self.I6_prime, self.I8_prime)
            self.I4_prime_prime = self.model3_1(B3, self.I3_prime, self.I5_prime, B5)
            self.I6_prime_prime = self.model3_2(B5, self.I5_prime, self.I7_prime, B7)
            self.I5_prime_prime = self.model4_1(self.I4_prime, self.I4_prime_prime, self.I6_prime_prime, self.I6_prime)

        return self.I2_prime, self.I4_prime, self.I6_prime, self.I8_prime, \
               self.I3_prime, self.I5_prime, self.I7_prime, \
               self.I4_prime_prime, self.I6_prime_prime, self.I5_prime_prime


class RDN_residual_interp_5_input_ConvLSTM_L(nn.Module):
    def __init__(self, modelType='lstm'):
        super(RDN_residual_interp_5_input_ConvLSTM_L, self).__init__()
        self.modelType = modelType
        self.clstm_4_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.clstm_6_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.clstm_8_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.clstm_5_prime_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.clstm_7_prime_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.clstm_6_prime_prime_prime = ConvLSTMCell(3, 3)  # 3 @ H x W
        self.model = RDN_residual_interp_5_input(lstm=True,GO=96,D=12)
        self.prev_state = None
        self.hidden_state = None

    def forward(self, B1, B3, B5, B7, B9, B11):
        pre_state_4_prime = None
        pre_state_6_prime = None
        pre_state_8_prime = None
        pre_state_5_prime_prime = None
        pre_state_7_prime_prime = None
        pre_state_6_prime_prime_prime = None
        hidden_state_4_prime = None
        hidden_state_6_prime = None
        hidden_state_8_prime = None
        hidden_state_5_prime_prime = None
        hidden_state_7_prime_prime = None
        hidden_state_6_prime_prime_prime = None
        input1 = [B1, B3, B5, B7, B9]
        input2 = [B3, B5, B7, B9, B11]
        res = []
        for i in [input1, input2]:
            B1, B3, B5, B7, B9 = i
            previous_inputs = [hidden_state_4_prime, hidden_state_6_prime, hidden_state_8_prime,
                               hidden_state_5_prime_prime, hidden_state_7_prime_prime, hidden_state_6_prime_prime_prime]
            self.Ft_p_1 = self.model(B1, B3, B5, B7, B9, previous_inputs)
            hidden_state_4_prime = self.Ft_p_1[1]
            hidden_state_6_prime = self.Ft_p_1[2]
            hidden_state_8_prime = self.Ft_p_1[3]
            hidden_state_5_prime_prime = self.Ft_p_1[5]
            hidden_state_7_prime_prime = self.Ft_p_1[6]
            hidden_state_6_prime_prime_prime = self.Ft_p_1[8]
            if self.modelType == 'lstm':
                # hidden_state, prev_state = self.clstm(hidden_state, prev_state)  # S x 128 @ H/4 x W/4
                hidden_state_4_prime, pre_state_4_prime = self.clstm_4_prime(hidden_state_4_prime, pre_state_4_prime)
                hidden_state_6_prime, pre_state_6_prime = self.clstm_6_prime(hidden_state_6_prime, pre_state_6_prime)
                hidden_state_8_prime, pre_state_8_prime = self.clstm_8_prime(hidden_state_8_prime, pre_state_8_prime)
                hidden_state_5_prime_prime, pre_state_5_prime_prime = self.clstm_5_prime_prime(hidden_state_5_prime_prime, pre_state_5_prime_prime)
                hidden_state_7_prime_prime, pre_state_7_prime_prime = self.clstm_7_prime_prime(hidden_state_7_prime_prime, pre_state_7_prime_prime)
                hidden_state_6_prime_prime_prime, pre_state_6_prime_prime_prime = self.clstm_6_prime_prime_prime(hidden_state_6_prime_prime_prime, pre_state_6_prime_prime_prime)
            else:
                pass
            res.append(self.Ft_p_1)

        return res[0][0], res[0][1], res[0][2], res[0][3], \
                res[0][4], res[0][5], res[0][6], \
                res[0][7], res[0][8], \
                res[0][9], \
               res[1][3], res[1][6], res[1][8], res[1][9]



def bin_stage4_lstm():
    model = RDN_residual_interp_5_input_ConvLSTM_L()
    return model
