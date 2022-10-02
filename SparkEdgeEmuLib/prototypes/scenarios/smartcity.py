from srds import ParameterizedDistribution

from SparkEdgeEmuLib.prototypes import nodes
from ether.blocks.cells import LANCell, IoTComputeBox, MobileConnection, FiberToExchange, counters
from ether.cell import GeoCell, SharedLinkCell
from ether.topology import Topology
import random

default_num_cells = 3
default_cloudlet_size = (5, 2)
default_cell_density = 2


class Cloudlet(LANCell):
    def __init__(self, server_node=nodes.cloudlet_vm, server_per_rack=5, racks=1, backhaul=None) -> None:
        self.racks = racks
        self.server_per_rack = server_per_rack
        self.server_node = server_node

        nodes = [self._create_rack] * racks

        super().__init__(nodes, backhaul=backhaul)

    def _create_identity(self):
        self.nr = next(counters['cloudlet'])
        self.name = 'cloudlet_%d' % self.nr
        self.switch = 'switch_%s' % self.name

    def _create_rack(self):
        return LANCell([self.server_node] * self.server_per_rack, backhaul=self.switch)

class SmartCity:
    def __init__(self, 
                 num_of_regions=default_num_cells, 
                 num_of_devices_per_region=3,
                 cloudlets = nodes.cloudlet_vm,
                 cloudlet_server_per_rack = 1,
                 cloudlet_number_of_racks = 1,
                 edge_devices=[nodes.nuc, nodes.rpi3b],
                 internet='internet',
                 edge_connection = '5G'
                ) -> None:
        """
        The UrbanSensingScenario builds on ideas from the Array of Things project, but extends it with proximate compute
        resources and adds a cloudlet to the city.

        The city is divided into cells, e.g., neighborhoods, and each cell has multiple urban sensing nodes and
        proximate compute resources. The devices in a cell are connected via a shared link. The city also hosts a
        cloudlet composed of server computers.

        The high-level parameters are: the number of cells, the cell density (number of nodes per cell), and the
        cloudlet size.

        :param num_cells: the number of cells to create, e.g., the neighborhoods in a city
        :param cell_density: the distribution describing the number of nodes in each neighborhood
        :param cloudlet_size: a tuple describing the number of servers in each rack, and the number of racks
        :param internet: the internet backbone that's being connected to (see `inet` package)
        """
        self.num_of_regions = num_of_regions
        self.cell_density = ParameterizedDistribution.lognorm((0.82, num_of_devices_per_region))
        self.cloudlet_size = (cloudlet_server_per_rack, cloudlet_number_of_racks)
        self.internet = internet
        self.edge_devices = edge_devices
        self.cloudlets = cloudlets

    def materialize(self, topology: Topology):
        topology.add(self.create_city())
        topology.add(self.create_cloudlet())

    def create_city(self) -> GeoCell:
                
        neighborhood = lambda size: SharedLinkCell(
            nodes=random.choices(self.edge_devices, k=size),
            shared_bandwidth=500,
            backhaul=MobileConnection(self.internet)
        )

        city = GeoCell(self.num_of_regions, nodes=[neighborhood], density=self.cell_density)

        return city

    def create_cloudlet(self) -> Cloudlet:
        return Cloudlet(self.cloudlets, *self.cloudlet_size, backhaul=FiberToExchange(self.internet))