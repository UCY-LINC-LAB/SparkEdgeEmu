# SparkEdgeEmu
*Interactive Performance Evaluation of Distributed Analytic Queries via Edge Emulation*

Edge Computing is promoted as a stable and efficient solution for IoT data processing and analytics. 
With Big Data Distributed Engines to be deployed on Edge infrastructures, data scientists seek solutions to evaluate the performance of their analytics queries. 
In this work, we propose SparkEdgeEmu, an interactive framework designed for data scientists in need of inspecting the performance of Spark analytic jobs without the Edge topology setup burden. 
SparkEdgeEmu provides: (i) parameterizable template-based use-cases for Edge infrastructures, (ii) real-time emulated environments serving ready-to-use Spark clusters, 
(iii) a unified and interactive programming interface for the framework's execution and query submission, and (vi) utilization metrics from the underlying emulated topology as well as performance and quantitative metrics from the deployed queries. We extensively evaluate the usability of our framework through a smart city use-case and we extracted useful performance hints for the queries' execution.


## Installation

Before starting the emulation, users need to create a docker network, namely `ether_net`, by executing the following command:

```shell script
docker network create ether_net
```

Then, users have to introduce some initial emulation parameters in `.env` file. An example of such parameters exists at `.env.example` file.

Finally, they execute `docker-compose up` command for emulator starting.

## Modeling Abstractions

When the framework is started, the users need to introduce their usecase model in a YAML file such as `usecase.yaml`.
In this file users describe the devices' types, connection types, use-case, and its parameters. 
For device types users can introduce also predefined devices such as `nuc`, `nx`, `nano`, `rpi3b`, `rpi3b_plus`, 
`cloudlet_vm`,  `rpi4_2G`, `rpi4_4G`, and `rpi4_8G`. 

```yaml
infrastructure:
    devices_types:
    - name: small-vm
      processor:
        cores: 4
        clock_speed: 1.5GHz
      memory: 4G
      disk:
        size: 32GB
        read: 95MB/s
        write: 90MB/s
    connection_types:
    - name: 5G
      downlink:
        data_rate: 90MBps
        latency: 2ms
        error_rate: 0.1%
      uplink: 
        data_rate: 90MBps
        latency: 2ms
        error_rate: 0.1%
usecase:
    usecase_type: smart_city
    parameters:
        num_of_regions: 1
        num_of_devices_per_region: 3
        cloudlet_server_per_rack: 1
        cloudlet_number_of_racks: 1
        edge_devices: [rpi3b, rpi4_2G]
        edge_connection: 5G
        cloudlets: small-vm

``` 

## Deployment and Execution

For the deployment, users import and instantiate the `EmulatorConnector` with the usecase file. 
Via the `connector` object, users can `deploy` the emulated infrastructure, create a spark session `create_spark_session`, 
and capture the performance metrics and duration of spark code execution (`with connector.timer()`).
After code execution, users retrieve the execution's metrics (`get_metrics()`) of each node both infrastructure (e.g., `cpu_util`) and spark-related metrics (e.g., `tasks`)

```python
from SparkEdgeEmuLib.connector import EmulatorConnector

connector = EmulatorConnector(usecase='usecase.yaml')
connector.deploy()

spark = connector.create_spark_session("evaluation-program")

with connector.timer():
    for i in range(10):
        df = spark.read.parquet("/data/*")
        df.groupBy("DOLocationID").agg({'driver_pay':'avg'}).collect()
        df.groupby('Hvfhs_license_num').agg({'*': 'count'}).collect()
        df.agg({'tips': 'sum'}).collect()

res = connector.get_metrics()

res['rpi3-b-0'].cpu_util.plot()
res['rpi3-b-0'].tasks.plot()
connector.undeploy()
```

