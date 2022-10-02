import itertools
from collections import defaultdict

from ether.cell import LANCell

from SparkEdgeEmuLib.prototypes.nodes import cloudlet_vm

counters = defaultdict(lambda: itertools.count(0, 1))

class VirtualizedCloudlet(LANCell):
    def __init__(self, num_of_vms=5, backhaul=None) -> None:
        self.num_of_vms = num_of_vms

        nodes = [self._create_rack]

        super().__init__(nodes, backhaul=backhaul)

    def _create_identity(self):
        self.nr = next(counters['cloudlet'])
        self.name = 'virtual_cloudlet_%d' % self.nr
        self.switch = 'switch_%s' % self.name

    def _create_rack(self):
        return LANCell([cloudlet_vm] * self.num_of_vms, backhaul=self.switch)