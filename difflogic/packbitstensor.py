import difflogic_cuda
import torch
import numpy as np


class PackBitsTensor:
    # t 는 [B, in_dim]
    def __init__(self, t: torch.BoolTensor, bit_count=32, device='cuda'):

        assert len(t.shape) == 2, t.shape

        self.bit_count = bit_count
        self.device = device

        if device == 'cuda':
            t = t.to(device).T.contiguous()     # transpose 해서 [in_dim, B]
            # difflogic_cuda.tensor_packbits_cuda(...) 함수는 C++ 바인딩에서 노출된 함수
            # difflogic_kernel.cu 의 tensor_packbits_cuda_kernel(...) 커널 실행
            # self.t 는 dtype이 bitcount에 따라 int32 / int 64 등이 됨
            # shape : [in_dim, ceil(B / bit_count)] --> batch에 있는 데이터들을 int32, int64로 묶음
            # self.pad_len은 패딩된 비트 수 이다.
            self.t, self.pad_len = difflogic_cuda.tensor_packbits_cuda(t, self.bit_count)
        else:
            raise NotImplementedError(device)

    def group_sum(self, k):
        assert self.device == 'cuda', self.device
        return difflogic_cuda.groupbitsum(self.t, self.pad_len, k)

    # flatten()을 override 해서 그냥 자기 자신을 반환한다.
    def flatten(self, start_dim=0, end_dim=-1, **kwargs):
        """
        Returns the PackBitsTensor object itself.
        Arguments are ignored.
        """
        return self

    def _get_member_repr(self, member):
        if len(member) <= 4:
            result = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member]
            return ' '.join(result)
        first_three = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member[:3]]
        sep = "..."
        final = np.binary_repr(member[-1], width=self.bit_count)[::-1]
        return f"{' '.join(first_three)} {sep} {final}"
    
    def __repr__(self):
        return '\n'.join([self._get_member_repr(item) for item in self.t])