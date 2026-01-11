import argparse
import math
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from .results_json import ResultsJSON

from . import mnist_dataset, uci_datasets

from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet

from dataclasses import replace, dataclass
from typing import Optional, Tuple, Dict, List
import time

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


def load_dataset(args):
    validation_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset == 'mnist20x20')
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset == 'mnist20x20')

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]


def get_model(args):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []

    arch = args.architecture    # architecture는 randomly_connected 하나만 존재한다.
    k = args.num_neurons
    l = args.num_layers

    ####################################################################################################################

    if arch == 'randomly_connected':
        # torch.nn.Flatten() 은 0번 차원(배치 차원)은 건드리지 않고, 1번 차원부터 마지막 차원까지를 하나로 합친다.
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw, id = 0))
        for i in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw, id=i+1))

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################

    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    print(f'total_num_weights={total_num_weights}')
    if args.experiment_id is not None:
        results.store_results({
            'total_num_neurons': total_num_neurons,
            'total_num_weights': total_num_weights,
        })

    model = model.to('cuda')

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to('cuda').round()).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()    # 모델을 eval 모드로 전환
        res = np.mean(
            [
                # model(PackBitsTensor(...)) 을 호출하면 torch.nn.Sequential 안에 있는 모듈들을 순서대로 실행한다.
                (model(PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    'cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

# -----------------------------
# Benchmark (speed)
# -----------------------------
@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1024,
    iters: int = 50,
    warmup: int = 10,
    input_size: Tuple[int, int, int] = (9, 32, 32),
) -> Tuple[float, float]:
    """
    반환:
      - avg_latency_ms (한 iteration에서 batch 1회 forward)
      - throughput (images/sec)
    """
    model.eval()
    x = torch.randn(batch_size, *input_size, device=device)
    x = x.to('cuda').reshape(x.shape[0], -1).round().bool()

    # warmup
    for _ in range(warmup):
        model(PackBitsTensor(x))
    if device.type == "cuda":
        torch.cuda.synchronize()

    
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(PackBitsTensor(x))
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
        else:
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    thr = batch_size * 1000 / avg_ms
    return avg_ms, thr

def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def count_buffers(model: nn.Module) -> int:
    sd_keys = model.state_dict().keys()
    return sum(b.numel() for name, b in model.named_buffers() if name in sd_keys)


def param_bytes(model: nn.Module) -> int:
    # 실제 dtype(half/float) 기준 파라미터 메모리
    return sum(p.numel() * p.element_size() for p in model.parameters())

def buffer_bytes(model: nn.Module) -> int:
    sd_keys = model.state_dict().keys()
    return sum(b.numel() * b.element_size() for name, b in model.named_buffers() if name in sd_keys)

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # ---------- Benchmark ----------
    bench_batch: int = 1024
    bench_iters: int = 50
    bench_warmup: int = 10


# 전반적인 구조/동작 흐름
# (1) 데이터셋 로딩 --> (2) LGN 모델 구성 --> (3) Adam으로 반복 학습 --> (4) 주기적으로 정확도 평가 --> (5) 옵션으로 PackBits 추론/CPU C 컴파일 추론
if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)

    args = parser.parse_args()

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # (1) 데이터 셋 로딩
    train_loader, validation_loader, test_loader = load_dataset(args)
    # (2) 모델 구성
    model, loss_fn, optim = get_model(args)

    ####################################################################################################################

    best_acc = 0

    # tqdm은 학습 진행 상황을 보여주는 진행 바를 생성한다.
    # x, y는 각각 입력 데이터와 정답 데이터를 나타낸다.
    for i, (x, y) in tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='iteration',
            total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
        y = y.to('cuda')

        # (3) Adam으로 반복 학습
        loss = train(model, x, y, loss_fn, optim)

        # (4) 주기적으로 정확도 평가
        if (i+1) % args.eval_freq == 0:
            if args.extensive_eval:
                train_accuracy_train_mode = eval(model, train_loader, mode=True)
                valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
            else:
                train_accuracy_train_mode = -1
                valid_accuracy_eval_mode = -1
                valid_accuracy_train_mode = -1
            train_accuracy_eval_mode = eval(model, train_loader, mode=False)
            test_accuracy_eval_mode = eval(model, test_loader, mode=False)
            test_accuracy_train_mode = eval(model, test_loader, mode=True)

            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'valid_acc_train_mode': valid_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
            }

            # PackBitsTensor 기반 초고속 추론
            if args.packbits_eval:
                r['train_acc_eval'] = packbits_eval(model, train_loader)
                r['valid_acc_eval'] = packbits_eval(model, train_loader)
                r['test_acc_eval'] = packbits_eval(model, test_loader)

            if args.experiment_id is not None:
                results.store_results(r)
            else:
                print(r)

            if valid_accuracy_eval_mode > best_acc:
                best_acc = valid_accuracy_eval_mode
                if args.experiment_id is not None:
                    results.store_final_results(r)
                else:
                    print('IS THE BEST UNTIL NOW.')

            if args.experiment_id is not None:
                results.save()

    ####################################################################################################################

    # (5) 옵션으로 PackBits 추론/CPU C 컴파일 추론
    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)

    cfg = CFG()
    device = torch.device('cuda')

    for layer in model:
        if isinstance(layer, LogicLayer):
            layer.removeParam()

    print("\n[LGN] Params:", f"{count_params(model):,}")
    print("[LGN] Buffers:", f"{count_buffers(model):,}")
    print("[LGN] Param bytes:", f"{bytes_to_mib(param_bytes(model)):.2f} MiB")
    print("[LGN] Buffer bytes:", f"{bytes_to_mib(buffer_bytes(model)):.2f} MiB")
    clgn_ms, clgn_thr = benchmark_inference(
        model, device,
        batch_size=cfg.bench_batch,
        iters=cfg.bench_iters,
        warmup=cfg.bench_warmup,
        input_size=(3*31,32,32)
    )
    print(f"[LGN] Inference: {clgn_ms:.3f} ms / batch({cfg.bench_batch}), {clgn_thr:.1f} img/s")

