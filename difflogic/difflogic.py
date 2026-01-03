import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor


########################################################################################################################

# 각 뉴런이 입력 2개만 받는 Logic Gate Layer
class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        # weights의 경우 각 뉴런이 16개의 게이트 중 어떤 걸 사용할지 나타내므로 (out_dim, 16) 의 모양이 된다.
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        # 연결은 학습하지 않고 초기화 후 고정된다.
        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        # assert 조건, 에러 메세지
        # x의 마지막 차원은 입력 차원과 같아야 한다.
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            print(self.indices[0].dtype, self.indices[1].dtype)

        # x[..., self.indices[0]] 에서 앞의 차원은 그대로 유지하고 마지막 차원에서만 인덱싱을 수행한다.
        # 예를 들어 x[..., indices] 가 x가 2차원이면 x[:, indices]와 같고 3차원이면 x[:,:, indices]와 같다.
        # self.indices[0] 는 크기가 out_dim 이고 각 원소는 0~in_dim-1 사이의 값이므로 이 각 값들을 뽑아 낸다고 생각하면 된다.
        # 그래서 a, b는 각각 output 뉴런에 들어갈 input 뉴런의 값들이 순서대로 저장되어 있다.
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            # self.weights.argmax(-1)는 마지막 차원에서 가장 큰 값이 있는 인덱스 번호를 찾는다.
            # self.weights의 크기가 (out_dim, 16)이면 각 행에서 가장 큰 값의 위치를 뽑아낸다.
            # 결과적으로 (out_dim,) 인 정수형 Long 텐서가 만들어진다.
            # torch.nn.functional.one_hot(..., 16) 은 위에서 구한 인덱스들을 길이가 16인 원-핫 벡터로 만든다. 
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    # PackBitsTensor를 사용한 초고속 추론
    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        # x.t 는 입력 데이터에 transpose를 취한 데이터이다.
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    # 뉴런 끼리의 connections를 만든다.
    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            # torch.randperm(n) := 0부터 n-1 까지의 숫자를 중복없이 무작위 순서로 배열한 1차원 텐서를 만든다.
            # % self.in_dim 연산을 하면 troch.randperm()으로 만들어진 각각의 원소들에 % self.in_dim 연산을 한다.
            c = torch.randperm(2 * self.out_dim) % self.in_dim  # 길이가 2 * out_dim 이고 모든 원소가 [0, in_dim-1] 범위의 1차원 텐서 생성
            c = torch.randperm(self.in_dim)[c]  # c를 다시 한 번 섞는다.
            c = c.reshape(2, self.out_dim)  # 크기가 1 x 2*out_dim 이었던 텐서를 2 x out_dim 크기의 2차원 텐서로 바꾼다.
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b    # a, b 각각은 크기가 out_dim 이고 원소의 크기는 [0, in_dim-1] 인 1차원 텐서이다.
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)


########################################################################################################################

# 마지막 LogicLayer가 만든 많은 출력 뉴런들을 클래스 수 k로 묶어서 합산해 점수를 만드는 집계 레이어
class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):   # 초고속 추론인 경우
            return x.group_sum(self.k)

        # 학습 / 일반 추론 경로
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        # x.shape[:-1]은 처음부터 마지막 차원 전까지를 의미한다 --> x의 shape이 (32,10,108) 이면 x.shape[:-1]는 (32, 10) 이 된다.
        # x의 마지막 차원이 (B, n) 이면 (B, k , group_size)로 바꾼다. k는 클래스의 크기.
        # .sum(-1)을 해서 그룹별 합을 한다. --> (B, k) 로 바뀜
        # tau 로 스케일 조정
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################
