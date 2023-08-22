import time
from string import Template
import requests
import networkx as nx
import numpy as np
import yaml
from FogifySDK import FogifySDK
from ether.core import Node, Link
from ether.topology import Topology
from ether.util import parse_size_string
from ether.util import to_size_string
from prometheus_pandas.query import Prometheus
from pyspark.sql import SparkSession

from SparkEdgeEmuLib.prototypes import nodes
from SparkEdgeEmuLib.prototypes.nodes import CapacityWithFreq
from SparkEdgeEmuLib.prototypes.scenarios.industrialiot import IndustrialIoTScenario
from SparkEdgeEmuLib.prototypes.scenarios.smartcity import SmartCity


class Placement(object):
    """
    A small class for placement policy definition.
    The services (services names) should be described in the initial docker-compose file.
    """

    def __init__(self):
        self.topology = dict()

    def deploy_service_to_node(self, service_name: str, node: Node):
        self.topology[node._name] = service_name

    def get_nodes(self) -> list:
        return [node_name for node_name in self.topology]


def template_creation(topology: Topology, controller_url: str = "http://contorller:5000") -> FogifySDK:
    """
    This function fulfills the templates of spark workers and returns an initialized Fogify object
    :param topology: The Ether topology object
    :param controller_url: Fogify's controller url
    :return:
    """
    filename = "spark-docker-compose.yaml"

    res = """
    version: '3.7'
    services:
    """

    worker_template = """
      $workername:
        image: andreper/spark-worker:3.0.0
        environment:
          - SPARK_WORKER_CORES=$cores
          - SPARK_WORKER_MEMORY=$memory
          - SPARK_MASTER_HOST=spark-master
          - SPARK_LOCAL_HOSTNAME=$workername
        volumes:
          - /data:/data
        command: $command
    """

    for node in topology.get_nodes():
        res += Template(worker_template).substitute(workername=str(node), cores=node.capacity.cpus,
            memory=f"{int(node.capacity.memory.real * 1e-6)}m", command="""sh -c "sleep 20 && bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077 >> logs/spark-worker.out" """)

    with open(filename, "w") as f:
        f.write(res)
    fogify = FogifySDK(controller_url, filename)
    return fogify


def topology_to_fogify(topology: Topology, fogify: FogifySDK, placement: Placement) -> FogifySDK:
    """
    Generates the final Fogify model
    :param topology: The Ether Topology
    :param fogify: An initialized Fogify object
    :param placement: Placement object that dictates how the services will be placed on the topology
    :return: The final Fogify object
    """
    #  Node translation from ether to Fogify model
    for n in topology.get_nodes():
        if n._name in placement.get_nodes():
            fogify.add_node(n._name, cpu_cores=n.capacity.cpus, cpu_freq=n.capacity.freq,  # TODO update the frequency
                            memory=to_size_string(n.capacity.memory,
                                                  'G'))  # the limit of memory is 7 due to the PC's limitation power

    # cloud properties
    cloud_latency = 100
    cloud_bandwidth = 100
    fogify.add_bidirectional_network("edge_net", bidirectional={'latency': {'delay': f'{cloud_latency}ms', },
        'bandwidth': f'{cloud_bandwidth}Mbps'

        })  # Maybe we need to describe the general network characteristics

    for n in topology.get_nodes():
        for j in topology.get_nodes():
            if type(n) == Node and type(
                    j) == Node and n != j and n._name in placement.get_nodes() and j._name in placement.get_nodes():  # introduce link connection between compute nodes
                bandwidth = min([k.bandwidth for k in topology.route(n, j).hops])
                latency = round(float(topology.route(n, j).rtt / 2), 2)
                fogify.add_link("edge_net", from_node=n._name, to_node=j._name, bidirectional=False, parameters={
                    'properties': {'latency': {'delay': f'{latency}ms', }, 'bandwidth': f'{bandwidth}Mbps'}})
    for node_name in placement.topology:
        fogify.add_topology_node(str(node_name), str(placement.topology[node_name]),
            # How can we introduce services in ether?
            str(node_name), networks=["edge_net"])

    return fogify


def draw_basic(topology, with_links=False):
    """
    Generates a Graph plot that illustrates the topology
    :param topology: The Ether topology
    :param with_links: If it is true, the graph has also the names of the links on them
    :return: None
    """
    pos = nx.kamada_kawai_layout(topology)  # positions for all nodes

    hosts = [node for node in topology.nodes if isinstance(node, Node)]
    links = [node for node in topology.nodes if isinstance(node, Link)]
    switches = [node for node in topology.nodes if str(node).startswith('switch_')]

    nx.draw_networkx_nodes(topology, pos, nodelist=hosts, node_color='silver', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(topology, pos, nodelist=links, node_color='g', node_size=50, alpha=0.9)
    nx.draw_networkx_nodes(topology, pos, nodelist=switches, node_color='y', node_size=200, alpha=0.8)
    nx.draw_networkx_nodes(topology, pos, nodelist=[node for node in topology.nodes if
                                                    isinstance(node, str) and node.startswith('internet')],
                           node_color='r', node_size=800, alpha=0.8)

    nx.draw_networkx_edges(topology, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(topology, pos, dict(zip(hosts, hosts)), font_size=13, verticalalignment='top')
    if with_links:
        nx.draw_networkx_labels(topology, pos, dict(zip(links, [l.tags['type'] for l in links])), font_size=10)


from datetime import datetime


class Timer:
    """
    The class that capsures the starting and ending time of execution
    """
    start = None
    end = None

    def __enter__(self):
        Timer.start = datetime.now()
        time.sleep(10)  # 10 secs delay
        return self

    def __exit__(self, *args):
        time.sleep(10) # 10 secs delay
        Timer.end = datetime.now()


# Metrics from Apache Spark
query_translate = dict(rdd_blocks="metrics_executor_rddBlocks",
    duration_seconds="metrics_executor_totalDuration_seconds_total",
    gctime="metrics_executor_totalGCTime_seconds_total", input_bytes="metrics_executor_totalInputBytes_bytes_total",
    shuffle_read="metrics_executor_totalShuffleRead_bytes_total",
    shuffle_write="metrics_executor_totalShuffleWrite_bytes_total",
    on_heap_storage_memory="metrics_executor_totalOnHeapStorageMemory_bytes",
    off_heap_storage_memory="metrics_executor_totalOffHeapStorageMemory_bytes",
    tasks="metrics_executor_totalTasks_total", used_memory="metrics_executor_memoryUsed_bytes")


# Available Use-cases
usecases = dict(smart_city=SmartCity, industrial_iot=IndustrialIoTScenario)


def scenario_definition(model: dict) -> Topology:
    """
    This function translates the model into a Ether use-case
    :param model: The parsed model
    :return: The generated Ether Topology
    """
    topology = Topology()
    infrastructure_nodes = model.get('infrastructure', {}).get('devices_types', [])
    ether_nodes = {}
    for node in infrastructure_nodes:
        name = node.get('name')
        cpus = int(node.get('processor').get('cores'))
        freq = 1000 * float(node.get('processor').get('clock_speed').lower().replace("ghz", ""))
        mem = node.get('memory')
        capacity = CapacityWithFreq(cpus=cpus, freq=freq, memory=parse_size_string(mem))
        labels = {}
        ether_nodes[name] = Node(name, capacity=capacity, arch='x86', labels=labels)
    ether_connection_types = []
    connection_types = model.get('infrastructure', {}).get('connection_types', [])

    # TODO ADD EDGE CONNECTION
    for connection in connection_types:
        pass

    usecase_model = model.get('usecase')
    usecase_type = usecase_model.get('usecase_type')
    usecase_parameters = usecase_model.get('parameters')
    usecase_class = usecases[usecase_type]

    input_devices = []
    for device in usecase_parameters.get('edge_devices'):
        cur_device = ether_nodes.get(device)
        if cur_device is None:
            cur_device = getattr(nodes, device)
        if cur_device is not None:
            input_devices.append(cur_device)

    cloudlet_type = usecase_parameters.get('cloudlets')
    cloudlet = ether_nodes.get(cloudlet_type)
    if cloudlet is None:
        cloudlet = getattr(nodes, cloudlet_type)

    usecase_parameters['edge_devices'] = input_devices
    usecase_parameters['cloudlets'] = cloudlet

    scenario = usecase_class(**usecase_parameters)
    scenario.materialize(topology)
    return topology


class EmulatorConnector:
    spark = None
    fogify = None
    scenario = None

    def __init__(self, controller_ip="controller", usecase="usecase.yaml"):
        self.controller_url = f"http://{controller_ip}:5000"
        self.usecase = {}
        with open(usecase, "r") as stream:
            try:
                self.usecase = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.topology = scenario_definition(self.usecase)
        self.apply_placement()

    def apply_placement(self):
        """
        It performs the placement of Spark services on the Topology and returns the final Fogify object
        :return: The generated Fogify object
        """
        fogify = template_creation(self.topology, controller_url=self.controller_url)
        placement = Placement()
        for node in self.topology.get_nodes():
            placement.deploy_service_to_node(node, node)
        self.fogify = topology_to_fogify(self.topology, fogify, placement)

    def create_spark_session(self, app_name, configs={}):
        """
        This function interconnects the emulation with Spark cluster master
        :param app_name: The name of the connection/application that will be visible on Spark master
        :param configs: Other configurations related to Spark cluster
        :return: A spark session object
        """
        spark = SparkSession.builder.appName(app_name).master("spark://spark-master:7077").config(
            "spark.ui.prometheus.enabled", True)
        for key, value in configs.items():
            spark.config(key, value)

        self.spark = spark.getOrCreate()
        return self.spark

    @property
    def node_ids(self):
        return [node["label"] for node in self.fogify.topology] if self.fogify else []

    def deploy(self):
        """
        Performs the deployment and instantation of the emulation
        :return: None
        """
        if self.fogify is None:
            raise Exception("There is no added scenario")
        self.fogify.deploy()

    def undeploy(self):
        """
        Performs the undeployment and destroys the emulation
        :return: None
        """
        if self.fogify is None:
            raise Exception("There is no added scenario")
        if self.spark is not None:
            self.spark.stop()
        self.fogify.undeploy()

    def get_metrics(self):
        """
        Gathers the metrics from the monitoring system and merges them in a dataframe
        :return: Metric's dataframe
        """
        metrics = {}
        emulation_metrics = self.get_metrics_from_emulation()
        cluster_metrics = self.get_metrics_from_cluster()
        for node_id in self.node_ids:
            metrics[node_id] = emulation_metrics[node_id]

            if node_id in cluster_metrics:
                metrics[node_id] = metrics[node_id].merge(cluster_metrics[node_id], on='count')
        return metrics

    def get_metrics_from_emulation(self):
        """
        Returns the infrastructure metrics from the execution period of the queries
        :return: Emulated Infrastructure metrics
        """
        metrics = {}
        for node_id in self.node_ids:
            df = self.fogify.get_metrics_from(node_id)
            df = df[df.timestamp >= int(Timer.start.timestamp()) * 1000][
                df.timestamp <= int(Timer.end.timestamp()) * 1000]
            df["count"] = np.arange(start=0, step=1, stop=len(df))
            metrics[node_id] = df
        return metrics

    def draw_scenario(self):
        if self.topology is None:
            raise Exception("There is no added scenario")
        draw_basic(self.topology)

    @property
    def timer(self):
        return Timer

    def get_executors_information(self):
        """
        Generates the required information for retrieving metrics from the executors
        :return:
        """
        sc = self.spark.sparkContext
        u = sc.uiWebUrl + '/api/v1/applications/' + sc.applicationId + '/allexecutors'
        res = {}
        for i in requests.get(u, proxies={"http": "", "https": ""}).json():
            if i["id"] != "driver":
                splitted = i["hostPort"].split(":")
                res[i["id"]] = {"ip": splitted[0], "port": splitted[1]}
        return res

    def get_metrics_from_cluster(self):
        """
        Returns the spark-related metrics from the execution period of the queries
        :return: Spark-related metrics
        """
        df = None
        p = Prometheus(f'http://prometheus:9090')
        results = {}
        res = self.get_executors_information()
        for executor_id, executor_details in res.items():
            results[executor_details['ip']] = None
            for query_id, query in query_translate.items():
                t = p.query_range(f'{query}' + '{executor_id="' + executor_id + '"}', Timer.start, Timer.end, '5s')
                t.columns = [query_id]
                if results[executor_details['ip']] is None:
                    results[executor_details['ip']] = t
                else:
                    results[executor_details['ip']] = results[executor_details['ip']].join(t)
            results[executor_details['ip']]["count"] = np.arange(start=0, step=1,
                                                                 stop=len(results[executor_details['ip']]))
        return results

    def overall_execution_time(self):
        """
        Presents the overall execution time of the queries in seconds
        :return: Execution time in seconds
        """
        return (Timer.end - Timer.start).total_seconds()