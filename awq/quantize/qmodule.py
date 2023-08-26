import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

        self.do_torch_awq_kernel  = False
        self.has_init_torch_kernel = False

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):     
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        if self.do_torch_awq_kernel is True:
            # The input x reshaping essentially mean to make 3-d input into 2-d tensor (matrix), AWQ custom kernel use this convention.
            # since qweight, scales, qzeros are attributes of self, not passing/creating them in self.torch_kernel_forward
            # 8 is actually a constant at runtime, it will always be 32 (int32)/ w_bit, we assume the constant of 8 always
            out = self.torch_kernel_forward(x.reshape(-1, x.shape[-1]))
        else:
            out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )

    def init_torch_kernel(self):
        dev = self.qweight.device
        # buffer for unpacked (full shape) qweight and qzero in int32 
        # these are tranposed of pytorch shape. it is by authors' design. TODO: why?
        self.register_buffer('unpacked_qweight', torch.zeros((self.in_features, self.out_features), dtype=torch.int32, device=dev))
        self.register_buffer('unpacked_qzeros', torch.zeros((self.in_features // self.group_size, self.out_features), dtype=torch.int32, device=dev))        
        
        # buffer for dequantized weight in FP16
        # this follows torch convention, shape of [OC, IC]
        self.register_buffer('dqweight', torch.zeros((self.out_features, self.in_features), dtype=torch.float16, device=dev))
        # scales is already in float16

        if self.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7] #TODO (VS): why we need this order? is this related to lower level kernel implementation?
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")

        pack_num = 32//self.w_bit

        # unpack qweight into unpacked_qweight
        for col in range(self.qweight.shape[1]):
            for i in range(32//self.w_bit):
                self.unpacked_qweight[:, col*pack_num + order_map[i]] = (self.qweight[:, col] >> i*self.w_bit)  & 15 # why 15? if last 4 bit of int32 is 1 then it is 15 in decimal

        # unpack dqzeros into unpacked_qzeros
        for col in range(self.qzeros.shape[1]):
            for i in range(32//self.w_bit):
                self.unpacked_qzeros[:, col*pack_num + order_map[i]] = (self.qzeros[:, col] >> i*self.w_bit)  & 15 
        
        #TODO (VS): why they were transposed?
        self.unpacked_qweight = self.unpacked_qweight.T
        self.unpacked_qzeros = self.unpacked_qzeros.T
        # self.scales = self.scales.T # we dont touch this

        for ic in range(self.unpacked_qweight.shape[1]): # for every column
            self.dqweight[:, ic] = ( self.unpacked_qweight[:,ic].to(torch.float16) - self.unpacked_qzeros[:, ic // self.group_size].to(torch.float16) ) * self.scales.T[:, ic // self.group_size]

        self.has_init_torch_kernel = True

        # reduce gpu memory footprint
        self.unpacked_qweight = self.unpacked_qweight.cpu()
        self.unpacked_qzeros = self.unpacked_qzeros.cpu()

    def torch_kernel_forward(self, x):
        # what does this function do?
        # it is an alternative implementation of AWQ cuda kernel by just torch function
        # why we do this? because this implementation shows how to unpacked and dequantize the AWQ weight and kernel c1onvention 
        # What is not covered? it does not demonstrate the parallelization of AWQ cuda kernel, it only assume equivalence at the input and output of AWQ kernel

        if self.has_init_torch_kernel is False:
            self.init_torch_kernel()

        # e.g. F.linear(input, self.weight, self.bias)
        return torch.nn.functional.linear(x, self.dqweight, self.bias)
        # do we need to reshape? no, it will be handled at the caller
                