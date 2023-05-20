import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quant_cuda

class QuantBase():
    def _quantization(x, quant_grid):
        shape = x.shape
        quant_array = x.reshape(-1)
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
        
        self.w_up = self.args.w_up
        self.a_up = self.args.a_up
        self.w_low = self.args.w_low
        self.a_low = self.args.a_low

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.register_buffer('bit', torch.tensor(bit))
        self.register_buffer('has_inited_quant_para', torch.tensor(0.0))
        self.register_buffer('quant_grid', torch.ones(2**bit))
        self.register_buffer('outliers', torch.ones(2**bit))
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

    @torch.no_grad()
    def int_value(self):
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
                
        values = torch.tensor(values, device=self.quant_grid.device) 
        values, _ = torch.sort(values)
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** B)

        return values

    @torch.no_grad()
    def flint_value(self,  exp_base = 0):
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

        ## exponent negtive
        for i in range(0, neg_exp_num + 1):
            exp_bit = i + 2
            exp_value = -(exp_bit - 1)
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** exp_value * (1 + 2 ** (-mant_bit) * j)
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
            
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** exp_max)

        return values
    
    # abfloat 
    @torch.no_grad()
    def outlier_value(self, exp_bit = 2, exp_base = 5):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
            
        value_bit = B
        mant_bit = value_bit - exp_bit
        values = []
        
        for i in range(exp_base, exp_base + 2 ** exp_bit):
            for j in range(int(2 ** mant_bit)):
                if i == exp_base and j == 0:
                    continue

                v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
                    
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
                    
        return values

    @torch.no_grad()
    def mse_loss(self, quant_tensor, source_tensor, p=2.0, is_perchannel=True):
        if is_perchannel:
            mean_tensor =  (quant_tensor-source_tensor).abs().pow(p).view(quant_tensor.shape[0], -1).mean(-1).unsqueeze(1)
            return mean_tensor
        else:
            return (quant_tensor-source_tensor).abs().pow(p).mean()
        
    @torch.no_grad()
    def search_mse(self, tensor):
        if self.is_perchannel and (not self.is_input):
            if not self.args.no_outlier:
                mean = tensor.view(tensor.shape[0], -1).mean(dim=-1)
                std = tensor.view(tensor.shape[0], -1).std(dim=-1)
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max, _ = tensor.view(tensor.shape[0], -1).abs().max(1)
            x_max = x_max.unsqueeze(1)            
            best_score = torch.ones_like(x_max) * 1e10
            alpha = x_max.clone()
            base_alpha = x_max.clone()
            lb = int(self.w_low)
            ub = int(self.w_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)

                score = self.mse_loss(quant_tensor, tensor)
                alpha[score < best_score] = new_alpha[score < best_score]
                best_score[score < best_score] = score[score < best_score]
        else:        
            if not self.args.no_outlier:
                mean = tensor.mean()
                std = tensor.std()
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max = tensor.abs().max()
            best_score = 1e10
            alpha = x_max.clone()
            base_alpha = alpha.clone()            
            lb = int(self.a_low)
            ub = int(self.a_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)
                score = self.mse_loss(quant_tensor, tensor, p = 2, is_perchannel=False)
                if score < best_score:
                    best_score = score
                    alpha = new_alpha

        return best_score.sum(), alpha, (alpha / x_max).mean().item()

    @torch.no_grad()
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

        if "-flint" in mode:
            self.mode = 'flint'
            self.quant_grid.data = self.flint_value()
            best_score_flint, _, _ = self.search_mse(data)
            modes.append('flint')
            mse_list.append(best_score_flint.item())

        mse_list = np.array(mse_list)
        mse_idx = np.argsort(mse_list)
        self.mode = modes[mse_idx[0]]

    @torch.no_grad()
    def _init_quant_para(self, data, data_b):
        with torch.no_grad():                    
            if self.has_inited_quant_para == 0:
                self.update_signed(data)                
                self.outliers.data = self.outlier_value()

                if self.is_perchannel:
                    x_max = data.view(data.shape[0], -1).abs().max(1).values
                    self.alpha.data = x_max.unsqueeze(1)
                else:
                    self.alpha.data = data.abs().max()

                if self.bit > 6:
                    self.mode = 'int'
                    self.quant_grid.data = self.int_value()
                else:
                    if "ant-" in self.mode:
                        self.search_adaptive_numeric_type(data)

                if self.mode == "flint":
                    self.quant_grid.data = self.flint_value()
                elif self.mode == "int":
                    self.quant_grid.data = self.int_value()
                else:
                    raise RuntimeError("Unsupported mode: " + self.mode)
                
                _, self.alpha.data, alpha_ratio = self.search_mse(data)

                quant_data = self._forward(data)
                self.mse = self.mse_loss(quant_data, data, 2, is_perchannel=self.is_perchannel).mean()
                print(self.mode, end="\t")
                print("%d-bit \t %s," %(self.bit.item(), self.name))
                
                self.has_inited_quant_para.data = torch.ones_like(self.has_inited_quant_para)
         
    @torch.no_grad()   
    def _forward(self, data, display=False):
        scale = self.alpha / torch.max(self.quant_grid)
        
        if self.is_perchannel: 
            data = (data.view(data.shape[0], -1) / scale).view(data.shape)
        else:
            data = data / scale
            
        if not self.args.no_outlier:
            quant_grid = torch.cat((self.quant_grid, self.outliers), dim = 0)
        else:
            quant_grid = self.quant_grid
            
        quant_data = QuantBase.forward(data, quant_grid)
        shape = data.shape
        
        # Outlier Victim Pair Encoding
        if not self.args.no_outlier:
            quant_data = quant_data.view(-1)                
            mask = quant_data.abs() > 32
            victim_odd = torch.roll(mask, 1, -1)
            victim_odd[::2] = 0
            victim_even = torch.roll(mask & (~victim_odd), -1, -1)
            victim_even[1::2] = 0
            victim = victim_even | victim_odd
            quant_data = quant_data * (~victim)

        quant_data = quant_data.view(shape)
        tensor = (quant_data - data).detach() + data
        
        if self.is_perchannel:
            tensor = (tensor.view(tensor.shape[0], -1) * scale).view(data.shape)
        else:
            tensor = tensor * scale

        return tensor

    @torch.no_grad()
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

        self._init_quant_para(tensor, input_tensor)

        q_tensor = self._forward(tensor)

        return q_tensor    

class TensorQuantizer(Quantizer):
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, input_tensor = None):
        return self.tensor_forward(tensor, input_tensor)

class Conv1dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(Conv1dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True)

    def set_param(self, conv):

        self.nf = conv.nf
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
            
    def _conv_forward(self, x, weight):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        return self._conv_forward(input, weight)


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
        return F.linear(input, weight, self.bias)