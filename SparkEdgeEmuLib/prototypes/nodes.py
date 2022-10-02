import itertools
from collections import defaultdict
from typing import Dict

from ether.core import Node, Capacity
from ether.util import parse_size_string

counters = defaultdict(lambda: itertools.count(0, 1))


class CapacityWithFreq(Capacity):

    def __init__(self, cpus: int = 1, freq: int = 1000, memory: int = 1024 * 1024 * 1024):
        self.memory = memory
        self.cpu_millis = cpus * freq
        self.cpus = cpus
        self.freq = freq

def name(self):
    return self.name.replace("_", "-")

Node._name = property(name)
Node.__str__ = name
    

def create_cloudlet_vm_node(name=None) -> Node:
    name = name if name is not None else 'cloudlet_vm_%d' % next(counters['cloudlet_vm'])

    return create_node(name=name, cpus=4, freq=2600, arch='x86', mem='16Gi',
                       labels={'ether.edgerun.io/type': 'vm', 'ether.edgerun.io/model': 'vm'})


def create_nuc_node(name=None) -> Node:
    name = name if name is not None else 'nuc_%d' % next(counters['nuc'])

    return create_node(name=name, cpus=4, freq=2400, arch='x86', mem='8Gi',
                       labels={'ether.edgerun.io/type': 'sffc', 'ether.edgerun.io/model': 'nuci5'})


def create_rpi3_b_node(name=None) -> Node:
    name = name if name is not None else 'rpi3_b_%d' % next(counters['rpi3_b'])

    return create_node(name=name, arch='arm32v7', cpus=4, freq=1200, mem='1Gi',
                       labels={'ether.edgerun.io/type': 'sbc', 'ether.edgerun.io/model': 'rpi3b', })


def create_rpi3_b_plus_node(name=None) -> Node:
    name = name if name is not None else 'rpi3_b_%d' % next(counters['rpi3_b+'])

    return create_node(name=name, arch='arm32v7', cpus=4, freq=1400, mem='1Gi',
                       labels={'ether.edgerun.io/type': 'sbc', 'ether.edgerun.io/model': 'rpi3b+', })


def create_rpi4_8G_node(name=None) -> Node:
    name = name if name is not None else 'rpi4_8G_%d' % next(counters['rpi4_8Gi'])

    return create_node(name=name, arch='arm32v7', cpus=4, freq=1500, mem='8Gi',
                       labels={'ether.edgerun.io/type': 'sbc', 'ether.edgerun.io/model': 'rpi4_8Gi', })


def create_rpi4_2G_node(name=None) -> Node:
    name = name if name is not None else 'rpi4_2G_%d' % next(counters['rpi4_2Gi'])

    return create_node(name=name, arch='arm32v7', cpus=4, freq=1500, mem='2Gi',
                       labels={'ether.edgerun.io/type': 'sbc', 'ether.edgerun.io/model': 'rpi4_2Gi', })


def create_rpi4_4G_node(name=None) -> Node:
    name = name if name is not None else 'rpi4_4G_%d' % next(counters['rpi4_4Gi'])

    return create_node(name=name, arch='arm32v7', cpus=4, freq=1500, mem='4Gi',
                       labels={'ether.edgerun.io/type': 'sbc', 'ether.edgerun.io/model': 'rpi4_4Gi', })


def create_nano(name=None) -> Node:
    name = name if name is not None else 'nano_%d' % next(counters['nano'])

    return create_node(name=name, cpus=4, freq=1430, arch='aarch64', mem='4Gi',
                       labels={'ether.edgerun.io/type': 'embai', 'ether.edgerun.io/model': 'nvidia_jetson_nano',
                           'ether.edgerun.io/capabilities/cuda': '10',
                           'ether.edgerun.io/capabilities/gpu': 'maxwell', })


def create_nx(name=None) -> Node:
    name = name if name is not None else 'nx_%d' % next(counters['nx'])

    return create_node(name=name, cpus=6, freq=1400, arch='aarch64', mem='8Gi',
                       labels={'ether.edgerun.io/type': 'embai', 'ether.edgerun.io/model': 'nvidia_jetson_nx',
                           'ether.edgerun.io/capabilities/cuda': '10', 'ether.edgerun.io/capabilities/gpu': 'volta', })


def create_node(name: str, cpus: int, freq: int, mem: str, arch: str, labels: Dict[str, str]) -> Node:
    capacity = CapacityWithFreq(cpus=cpus, freq=freq, memory=parse_size_string(mem))
    return Node(name, capacity=capacity, arch=arch, labels=labels)



nuc = create_nuc_node
nx = create_nx
nano = create_nano

rpi3b = create_rpi3_b_node
rpi3b_plus = create_rpi3_b_plus_node

cloudlet_vm = create_cloudlet_vm_node

rpi4_8G = create_rpi4_8G_node
rpi4_2G = create_rpi4_2G_node
rpi4_4G = create_rpi4_4G_node
