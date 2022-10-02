from srds import ConstantSampler
from SparkEdgeEmuLib.prototypes import nodes
from ether.blocks.cells import IoTComputeBox, BusinessIsp
from ether.cell import LANCell, SharedLinkCell, UpDownLink
from ether.topology import Topology

from SparkEdgeEmuLib.prototypes.cells import VirtualizedCloudlet

default_num_cells = 1
default_cell_density = ConstantSampler(10)



class IndustrialIoTScenario:
    def __init__(self,
                 num_premises=default_num_cells,
                 premises_density=default_cell_density,
                 floor_compute=[nodes.nuc, nodes.rpi3b],
                 floor_iot=[nodes.rpi3b] * 3,
                 num_of_cloudlets=1,
                 internet='internet') -> None:
        """
        The IIoT scenarios with several factories, that have a factory floor with IoT devices and a on-premises managed
        cloudlet.

        :param num_premises: the number of premises, each premises is a factory with a floor and a cloudlet
        :param premises_density: currently not used, but the idea is that the total number of devices on a premises vary
        according to the parameter. but it's unclear how the total number of devices should be split among the floor and
        the cloudlet.
        :param internet:
        """
        self.num_premises = num_premises
        self.premises_density = premises_density
        self.internet = internet
        self.floor_compute = floor_compute
        self.floor_iot = floor_iot
        self.num_of_cloudlets = num_of_cloudlets

    def materialize(self, topology: Topology):
        for _ in range(self.num_premises):
            floor_compute = IoTComputeBox(nodes=self.floor_compute)
            floor_iot = SharedLinkCell(nodes=self.floor_iot)

            factory = LANCell([floor_compute, floor_iot], backhaul=BusinessIsp(self.internet))
            factory.materialize(topology)

            cloudlet = VirtualizedCloudlet(self.num_of_cloudlets, backhaul=UpDownLink(10000, 10000, backhaul=factory.switch))
            cloudlet.materialize(topology)
