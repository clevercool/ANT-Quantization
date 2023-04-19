import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import quant_cuda
import torch.distributed as dist
from quant_affine import *

class QuantBase():
    def _quantization(x, quant_grid):
        shape = x.shape
        quant_array = x.view(-1)
        quant_grid = quant_grid.type_as(quant_array)
        quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
        quant_array = quant_array.view(shape)
        return quant_array

    @staticmethod
    def forward(real_val, quant_grid):
        with torch.no_grad():
            dequantized_val = QuantBase._quantization(real_val, quant_grid)
            return dequantized_val


class Quantizer(nn.Module):
    def __init__(self, mode="base", bit=8, is_signed=True, is_enable=False, is_input=False, args=None, operator=None):
        super(Quantizer, self).__init__()
        self.mode = mode
        self.is_input = is_input
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_activation = is_enable
        self.is_enable_weight = is_enable
        self.args = args
        self.operator = operator

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.register_buffer('bit', torch.tensor(bit))
        self.register_buffer('has_inited_quant_para', torch.tensor(0.0))
        self.register_buffer('quant_grid', torch.ones(2**bit))
        
        self.w_up = self.args.w_up
        self.a_up = self.args.a_up
        self.w_low = self.args.w_low
        self.a_low = self.args.a_low

        self.percent = self.args.percent / 100
        self.is_perchannel = True
        if is_input:
            # Input shouldn't be per-channel quantizaton！
            self.is_perchannel = False
        self.search = args.search
        self.mse = torch.tensor(0.0)

        ## debug
        self.name = None

    def disable_input_quantization(self):
        self.is_enable_activation = False
        
    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def update_signed(self, tensor):
        if tensor.min() < 0:
            self.is_signed = True

    def convert_tensor(self, values):
        if 2 ** self.bit.item() > len(values):
            values.append(0.)
        assert(2 ** self.bit.item() == len(values))
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
        values = values.mul(10.0 / torch.max(values))
        # print(values.shape, values.data, end="--")
        return values

    def apot_value(self):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        base_a = [0.]
        base_b = [0.]
        base_c = [0.]
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass

        values = []
        for a in base_a:
            for b in base_b:
                for c in base_c:
                    values.append((a + b + c))
                    if self.is_signed:
                        values.append(-(a + b + c))
                    
        return self.convert_tensor(values)
    
    def float_value(self):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        exp_bit = 3
        man_bit = B - 3
        if B == 2:
            exp_bit = 2
            man_bit = 0
        values = []
        min_to_zero = True
        for i in range(2 ** exp_bit):
            for j in range(2 ** man_bit):
                if min_to_zero:
                    values.append(0.)
                    min_to_zero = False
                else:
                    values.append(2 ** i * (1 + j * 2 ** (-man_bit)))
                    if self.is_signed:
                        values.append(- 2 ** i * (1 + j * 2 ** (-man_bit)))

        return self.convert_tensor(values)


    def float_value(self, eb = 3):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        exp_bit = eb
        man_bit = B - exp_bit
        if B == 2:
            exp_bit = 2
            man_bit = 0
        values = []
        min_to_zero = True
        subnormal = True
        for i in range(2 ** exp_bit):
            for j in range(2 ** man_bit):
                if min_to_zero:
                    values.append(0.)
                    min_to_zero = False
                else:
                    if subnormal:
                        values.append((2 ** i) * (j * 2 ** (-man_bit)))
                    else:
                        values.append((2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))

                    if self.is_signed:
                        if subnormal:
                            values.append(-(2 ** i) * (j * 2 ** (-man_bit)))
                        else:
                            values.append(-(2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))
            subnormal = False

        return self.convert_tensor(values)

    def pot_value(self):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        exp_bit = B
        values = []
        values.append(0.)
        for i in range(0, 2 ** exp_bit - 1):
            values.append(2 ** i)
            if self.is_signed:
                values.append(-2 ** i)

        return self.convert_tensor(values)


    def int_value(self, q_type="int"):
        bit_width = self.bit.item()
        B = bit_width
        if self.is_signed:
            B = bit_width - 1

        values = []
        values.append(0.)
        for i in range(1, 2 ** B):
            values.append(i)
            if self.is_signed:
                values.append(-i)

        if q_type == "int":
            if self.is_signed:
                values.append(-2 ** B)

        return self.convert_tensor(values)

    def flint_value(self,  exp_base = 0):
        ################## Flint Representation #################
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        value_bit = B
        assert(value_bit >= 2)

        exp_num =     value_bit * 2 - 1
        neg_exp_num = value_bit - 1
        pos_exp_num = value_bit - 1
       
        
        exp_max = pos_exp_num + exp_base
        exp_min = -neg_exp_num

        ## zero
        values = [0.]

        # values = [0.]
        ## exponent negtive
        for i in range(0, neg_exp_num + 1):
            exp_bit = i + 2
            exp_value = -(exp_bit - 1)
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)

        ## exponent zero
        exp_bit = 2
        exp_value = 0
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if self.is_signed:
                values.append(-v)
        ## exponent positive     
        for i in range(1, pos_exp_num):
            exp_bit = i + 2
            exp_value = i
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
        ## max value
        values.append(2 ** exp_max)
        if self.is_signed:
            values.append(-2 ** exp_max)

        return self.convert_tensor(values)

    def mse_loss(self, quant_tensor, source_tensor, p=2.0, is_perchannel = True):
        if is_perchannel:
            mean_tensor =  (quant_tensor-source_tensor).abs().pow(p).view(quant_tensor.shape[0], -1).mean(-1).unsqueeze(1)
            return mean_tensor
        else:
            return (quant_tensor-source_tensor).abs().pow(p).mean()

    def search_mse(self, tensor):
        if self.is_perchannel and (not self.is_input):
            x_max, _ = tensor.view(tensor.shape[0], -1).abs().max(1)
            x_max = x_max.unsqueeze(1)            
            best_score = torch.ones_like(x_max) * 1e10
            
            alpha = x_max.clone()
            base_alpha = x_max.clone()
            lb = int(self.w_low)
            if self.bit > 6:
                lb = int(95)
            ub = int(self.w_up)
            for i in range(lb, ub):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)

                score = self.mse_loss(quant_tensor, tensor)
                alpha[score < best_score] = new_alpha[score < best_score]
                best_score[score < best_score] = score[score < best_score]
        else:        
            x_max = tensor.abs().max()
            best_score = 1e10
            alpha = x_max.clone()
            base_alpha = alpha.clone()
            
            lb = int(self.a_low)
            if self.bit > 6:
                lb = int(95)
            ub = int(self.a_up)
            for i in range(lb, ub):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)
                score = self.mse_loss(quant_tensor, tensor, p = 2, is_perchannel=False)
                if score < best_score:
                    best_score = score
                    alpha = new_alpha

        return best_score.sum(), alpha, (alpha / x_max).mean().item()

    def search_adaptive_numeric_type(self, data):
        modes = []
        mse_list = []
        mode = self.mode
        if "-int" in mode:
            self.mode = 'int'
            self.quant_grid.data = self.int_value()
            best_score_int, _, _ = self.search_mse(data)
            modes.append('int')
            mse_list.append(best_score_int.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, INT   score: %f" %best_score_int)
        
        if "-flint" in mode:
            self.mode = 'flint'
            self.quant_grid.data = self.flint_value()
            best_score_flint, _, _ = self.search_mse(data)
            modes.append('flint')
            mse_list.append(best_score_flint.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, Flint score: %f" %best_score_flint)
        
        if "-pot" in mode:
            self.mode = 'pot'
            self.quant_grid.data = self.pot_value()
            best_score_pot, _, _ = self.search_mse(data)
            modes.append('pot')
            mse_list.append(best_score_pot.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, POT   score: %f" %best_score_pot)

        if "-float" in mode:
            self.mode = 'float'
            self.quant_grid.data = self.float_value()
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT score: %f" %best_score_float)

        if "-float1" in mode:
            self.mode = 'float1'
            self.quant_grid.data = self.float_value(1)
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float1')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT 1 score: %f" %best_score_float)

        if "-float2" in mode:
            self.mode = 'float2'
            self.quant_grid.data = self.float_value(1)
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float2')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT 2 score: %f" %best_score_float)

        if "-float3" in mode:
            self.mode = 'float3'
            self.quant_grid.data = self.float_value(1)
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float3')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT 3 score: %f" %best_score_float)

        if "-float4" in mode:
            self.mode = 'float4'
            self.quant_grid.data = self.float_value(1)
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float4')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT 4 score: %f" %best_score_float)

        if "-apot" in mode:
            self.mode = 'apot'
            self.quant_grid.data = self.apot_value()
            best_score_apot, _, _ = self.search_mse(data)
            modes.append('apot')
            mse_list.append(best_score_apot.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, APOT score: %f" %best_score_apot)

        mse_list = np.array(mse_list)
        mse_idx = np.argsort(mse_list)
        self.mode = modes[mse_idx[0]]
    
    def outlier_set(self, data):
        def reduce_ave_tensor(tensor):
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= dist.get_world_size()
            return rt

        q = torch.tensor([self.percent], device = data.device)
        # self.percent_value_int4 = torch.quantile(data.abs().view(-1), q, dim=0)
        self.percent_value_int4 = torch.tensor(np.percentile(data.abs().cpu().numpy(),self.percent*100), device=data.device)
        self.percent_value_int16 = data.abs().max()

        self.percent_value_int4.data = reduce_ave_tensor(self.percent_value_int4.data)
        self.percent_value_int16.data = reduce_ave_tensor(self.percent_value_int16.data)

        if dist.get_rank() == 0: 
            print(self.name, self.percent_value_int4.item(), self.percent_value_int16.item())
        self.is_perchannel = False
        self.quant_grid.data = self.int_value()
        self.has_inited_quant_para.data = torch.ones_like(self.has_inited_quant_para)

    def outlier_quant(self, data):
        mask_int16 = data.abs() > self.percent_value_int4

        if self.percent_value_int4 > 0:
            scale = self.percent_value_int4 / torch.max(self.quant_grid)
            data_int4 = data / scale
            quant_data = QuantBase.forward(data_int4, self.quant_grid)
            tensor = quant_data.clone().detach()
            tensor =  tensor * scale
        else:
            tensor = data.clone().detach()

        if self.is_signed:
            level = 2**16 - 1
        else:
            level = 2**15 - 1
            
        if self.percent < 100:
            scale = (self.percent_value_int16 - self.percent_value_int4) / level
            data_int16 = data[mask_int16].abs()
            sign_int16 = data[mask_int16].sign()
            data_int16 = data_int16 - self.percent_value_int4
            quant_data = (data_int16 / scale).round() * scale
            quant_data = quant_data + self.percent_value_int4
            quant_data = quant_data * sign_int16
            tensor[mask_int16] = (quant_data - tensor[mask_int16]).detach() + tensor[mask_int16]

        return tensor


    def _init_quant_para(self, data, data_b):
        with torch.no_grad():                    
            if self.has_inited_quant_para == 0:
                self.update_signed(data)                

                if self.is_perchannel:
                    x_max = data.view(data.shape[0], -1).abs().max(1).values
                    self.alpha.data = x_max.unsqueeze(1)
                else:
                    self.alpha.data = data.abs().max()

                if self.mode == 'outlier':
                    return self.outlier_set(data)

                if self.bit > 6:
                    self.mode = 'int'
                else:
                    if "ant-" in self.mode:
                        self.search_adaptive_numeric_type(data)

                alpha_ratio = 1.0
                if self.mode == "flint":
                    self.quant_grid.data = self.flint_value()
                    # _, self.alpha.data, alpha_ratio = self.search_mse(data)
                elif self.mode == "int":
                    self.quant_grid.data = self.int_value()
                    # _, self.alpha.data, alpha_ratio = self.search_mse(data)
                elif self.mode == "pot":
                    self.quant_grid.data = self.pot_value()   
                    # _, self.alpha.data, alpha_ratio = self.search_mse(data)
                elif self.mode == "apot":
                    self.quant_grid.data = self.apot_value()
                elif self.mode == "float":
                    self.quant_grid.data = self.float_value()
                elif self.mode == "float1":
                    self.quant_grid.data = self.float_value(1)
                elif self.mode == "float2":
                    self.quant_grid.data = self.float_value(2)
                elif self.mode == "float3":
                    self.quant_grid.data = self.float_value(3)
                elif self.mode == "float4":
                    self.quant_grid.data = self.float_value(4)
                else:
                    raise RuntimeError("Unsupported mode: " + self.mode)
                
                _, self.alpha.data, alpha_ratio = self.search_mse(data)



                def reduce_ave_tensor(tensor):
                    rt = tensor.clone()
                    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                    rt /= dist.get_world_size()
                    return rt

                quant_data = self._forward(data)
                self.mse = self.mse_loss(quant_data, data, 2, is_perchannel=self.is_perchannel).mean()
                dist.broadcast(self.mse, 0)
                if dist.get_rank() == 0:
                    print(self.mode, end="\t")
                    print("%d-bit \t %s," %(self.bit.item(), self.name))
                
                self.alpha.data = reduce_ave_tensor(self.alpha.data)
                dist.broadcast(self.quant_grid, 0)

                self.has_inited_quant_para.data = torch.ones_like(self.has_inited_quant_para)
            
    def _forward(self, data):
        scale = self.alpha / torch.max(self.quant_grid)

        if self.is_perchannel:
            data = (data.view(data.shape[0], -1) / scale).view(data.shape)
        else:
            data = data / scale

        quant_data = QuantBase.forward(data, self.quant_grid)
        tensor = (quant_data - data).detach() + data

        if self.is_perchannel:
            tensor =  (tensor.view(tensor.shape[0], -1) * scale).view(data.shape)
        else:
            tensor =  tensor * scale

        return tensor
    
    def tensor_forward(self, tensor, input_tensor = None):
        if self.mode == "base":
            return tensor
        if not self.is_enable:
            return tensor
        if self.is_input:
            if not self.is_enable_activation:
                return tensor
        else:
            if not self.is_enable_weight:
                return tensor

        with torch.no_grad():
            self._init_quant_para(tensor, input_tensor)

        if self.mode == 'outlier':
            q_tensor = self.outlier_quant(tensor)
        else:
            q_tensor = self._forward(tensor)

        return q_tensor    

class TensorQuantizer(Quantizer):
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, input_tensor = None):
        return self.tensor_forward(tensor, input_tensor)

class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(Conv2dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        self.quant_weight.alpha.data = torch.ones([self.out_channels,1])

        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        # print("convolution", input.unique().numel(), self.quant_input.name)
        return self._conv_forward(input, weight)


class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(LinearQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=F.linear)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=F.linear, is_input=True)


    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_weight.alpha.data = torch.ones([self.out_features, 1])

        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        weight = self.quant_weight(self.weight, input) 
        input = self.quant_input(input, self.weight)
        # print(input.unique().numel(), self.quant_input.name)
        return F.linear(input, weight, self.bias)
