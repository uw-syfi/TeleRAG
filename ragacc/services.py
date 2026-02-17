import multiprocessing as mp
import subprocess
import zmq
import pickle
import torch
import asyncio
import os
import time
from dataclasses import dataclass

from .llm_serving import RAGAccLLM
from .index import RAGAccIndex
from .numa import numa_run_on_node
from .zmq_utils import async_send_recv

SHUTDOWN_REQUEST                                    = 0
TEST_START_REQUEST                                  = 1
LLM_GENERATE_SIM_REQUEST                            = 2
RETRIEVAL_PREFETCH_REQUEST                          = 3
RETRIEVAL_CLEAR_PREFETCH_DATA_REQUEST               = 4
RETRIEVAL_FIND_CLUSTERS_REQUEST                     = 5
RETRIEVAL_SEARCH_REQUEST                            = 6
RETRIEVAL_SWITCH_GPU_REQUEST                        = 7
RETRIEVAL_RESIZE_CACHE_REQUEST                      = 8
RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST    = 9
RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_REQUEST        = 10
RETRIEVAL_CHANGE_NUM_GPU_REQUEST                    = 11
RETRIVAL_CHANGE_CACHE_FRACTION_REQUEST              = 12
RAG_PIPELINE_EVALUATION_REQUEST                     = 13
RAG_WARM_UP_REQUEST                                 = 14
RAG_TXT_TO_EMB_REQUEST                              = 15
RAG_CLEAR_ALL_PREFETCH_DATA_REQUEST                 = 16
RAG_UPDATE_NPROBE_REQUEST                           = 17

@dataclass
class Request:
    type: int
    args: dict

REPLY_STATUS_OK = 0
REPLY_STATUS_ERROR = 1

@dataclass
class Reply:
    status: int
    data: dict|None = None


class Service:
    def __init__(self, args, port: int, byte_mode=False, stand_alone=False, numa_node=None):
        """
        Args:
            args: Arguments for the service.
            port (int): Port number for the service.
            byte_mode (bool): Whether to use byte mode for communication.
            stand_alone (bool): Whether the service runs in standalone mode.
                If True, it will not spawn a new process.
            numa_node (int): NUMA node to run the service on.
        """
        self.args = args
        self.port = port
        self.byte_mode = byte_mode
        self.numa_node = numa_node
        self.shutdown = False
        ctx = mp.get_context('spawn')
        if stand_alone:
            self.process = None
            self.init_process(args)
        else:
            self.process = ctx.Process(target=self.init_process, args=(args,))

    def __del__(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def init_process(self, args):
        # Initialize ZMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.REP)  # REP (Reply) socket for server
        socket.bind(f"tcp://*:{self.port}")  # Bind to port

        # Init the service
        if self.numa_node is not None:
            numa_run_on_node(self.numa_node)
        self.init_service(args)

        # Start receiving requests and responding
        while True:
            if self.shutdown:
                socket.close()
                return
            message = socket.recv()  # Receive a message
            if not self.byte_mode:
                message = pickle.loads(message)
            serve_request = self.serve_request(message)  # Process the request
            # try:
            #     serve_request = self.serve_request(message)  # Process the request
            # except Exception as e:
            #     print(f"Error processing request: {e}", file=sys.stderr)
            #     serve_request = Reply(status=REPLY_STATUS_ERROR, data={'error': str(e)})
            if not self.byte_mode:
                serve_request = pickle.dumps(serve_request)
            socket.send(serve_request)

    def init_service(self, args):
        """Init the service. This method should be overridden in subclasses."""
        raise NotImplementedError()

    def serve_request(self, data):
        """Serve a request. This method should be overridden in subclasses."""
        if data.type == SHUTDOWN_REQUEST:
            self.shutdown = True
            return Reply(status=REPLY_STATUS_OK)
        if data.type == TEST_START_REQUEST:
            return Reply(status=REPLY_STATUS_OK)
        return None

    def start(self):
        if self.process is None:
            raise ValueError("Standalone service don't need to start manually.")
        self.process.start()

        # Wait for the service to initialize
        # The service will respond only when it has finished initialization
        asyncio.run(wait_service_initialization(
            f"tcp://localhost:{self.port}",
        ))

    async def async_start(self):
        if self.process is None:
            raise ValueError("Standalone service don't need to start manually.")
        self.process.start()

        # Wait for the service to initialize
        # The service will respond only when it has finished initialization
        await async_send_recv(
            f"tcp://localhost:{self.port}",
            Request(type=TEST_START_REQUEST, args={}), byte_mode=False)

    def close(self):
        if self.process is None:
            raise ValueError("Standalone service don't need to close manually.")
        request = Request(type=SHUTDOWN_REQUEST, args={})
        asyncio.run(async_send_recv(
            f"tcp://localhost:{self.port}",
            request, byte_mode=False))
        self.process.terminate()

@dataclass
class ServiceInfo:
    addr: str
    port: int

class ServiceManager:
    def __init__(self):
        self.services = {}

    @staticmethod
    def service_key(service_type: str, id: int):
        """
        Generate a unique key for the service based on its type and ID.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
            id (int): ID of the service instance.
        Returns:
            str: Unique key for the service.
        """
        return f"{service_type}_{id}"

    def find_service(self, service_type: str, id: int) -> ServiceInfo | None:
        """
        Find a service by its type and ID.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
            id (int): ID of the service instance.
        Returns:
            Service: The service instance if found, otherwise None.
        """
        service_key = self.service_key(service_type, id)
        return self.services.get(service_key, None)

    def find_all_services(self, service_type: str) -> list[ServiceInfo]:
        """
        Find all services of a given type.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
        Returns:
            list[Service]: List of service instances of the given type.
        """
        result = []
        for key, service in self.services.items():
            if key.startswith(f"{service_type}_"):
                result.append(service)
        return result

    def get_service_address(self, service_type: str, id: int):
        """
        Get the address of a service by its type and ID.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
            id (int): ID of the service instance.
        Returns:
            str: The address of the service if found, otherwise None.
        """
        service = self.find_service(service_type, id)
        if service is not None:
            return service.addr
        return None

    def get_all_service_addresses(self, service_type: str) -> list[str]:
        """
        Get the addresses of all services of a given type.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
        Returns:
            list[str]: List of addresses of services of the given type.
        """
        services = self.find_all_services(service_type)
        return [service.addr for service in services]

    def register_service(self, service_type: str, service_id: int,
                         service_info: ServiceInfo):
        """
        Register a service with the service manager.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
            service_id (int): ID of the service instance.
            service_info (ServiceInfo): Information about the service to register.
        """
        service_key = self.service_key(service_type, service_id)
        self.services[service_key] = service_info

    def shutdown_all_matching_services(self, service_type: str):
        """
        Shutdown all services of a given type.
        Args:
            service_type (str): Type of the service (e.g., 'retrieval', 'llm', 'rag').
        """
        services = self.find_all_services(service_type)
        async def concurrent_shutdown():
            await asyncio.gather(*[
                async_send_recv(
                    service_info.addr,
                    Request(type=SHUTDOWN_REQUEST, args={}), byte_mode=False
                ) for service_info in services
            ])
        asyncio.run(concurrent_shutdown())


class RetrievalService(Service):
    def __init__(self, args, port: int, byte_mode=False, stand_alone=False, numa_node=None):
        super().__init__(args, port, byte_mode, stand_alone, numa_node)
        self.index = None
        self.device = torch.device('cpu')  # Default to CPU

    def init_service(self, args):
        # Initialize the retrieval service with the provided arguments
        print(f"Initializing retrieval service...")
        args.gpu_id = 0
        self.index = RAGAccIndex(args)
        self.device = torch.device(f"cuda:{args.gpu_id}")

    def serve_request(self, data):
        result = super().serve_request(data)
        if result is not None:
            return result
        # Process the request and return the result
        if data.type == RETRIEVAL_PREFETCH_REQUEST:
            self.index.prefetch_batch(
                data.args['prefetch_emb'].to(self.device), data.args['prefetch_budget'],
            )
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RETRIEVAL_CLEAR_PREFETCH_DATA_REQUEST:
            self.index.clear_prefetch_data()
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RETRIEVAL_FIND_CLUSTERS_REQUEST:
            clusters = self.index.find_clusters(
                data.args['emb'].to(self.device), data.args['nprobe'],
            )
            return Reply(status=REPLY_STATUS_OK, data={'clusters': clusters})
        elif data.type == RETRIEVAL_SEARCH_REQUEST:
            results = self.index.search(
                data.args['emb'].to(self.device), data.args['topk'], data.args['nprobe'],
                data.args['gpu_only_search'], data.args['cpu_only_search'],
                data.args['runtime_fetch'], data.args['fetch_emb'],
                data.args['fetch_nprobe'],
            )
            torch.cuda.synchronize()
            return Reply(status=REPLY_STATUS_OK, data={'results': results})
        elif data.type == RETRIEVAL_SWITCH_GPU_REQUEST:
            self.index.switch_gpu(data.args['gpu_id'], data.args['update_cache_record'])
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RETRIEVAL_RESIZE_CACHE_REQUEST:
            self.index.resize_cache_and_clear_for_next()
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST:
            overlap, total_count = self.index.get_cache_clusters_overlap_sim(
                data.args['emb'].to(self.device), data.args['nprobe'], data.args['gpu_id']
            )
            return Reply(status=REPLY_STATUS_OK, data={'overlap': overlap, 'total_count': total_count})
        elif data.type == RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_REQUEST:
            overlap, total_count = self.index.get_cache_clusters_overlap(
                data.args['emb'].to(self.device), data.args['nprobe']
            )
            return Reply(status=REPLY_STATUS_OK, data={'overlap': overlap, 'total_count': total_count})
        elif data.type == RETRIEVAL_CHANGE_NUM_GPU_REQUEST:
            self.index.change_num_gpu(data.args['num_gpu'])
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RETRIVAL_CHANGE_CACHE_FRACTION_REQUEST:
            self.index.change_cache_fraction(data.args['fraction'])
            return Reply(status=REPLY_STATUS_OK)
        raise ValueError(f"Unknown request type: {data.type}")


class LLMService(Service):
    def __init__(self, args, port: int, byte_mode=False, stand_alone=False, numa_node=None):
        super().__init__(args, port, byte_mode, stand_alone, numa_node)
        self.llm = None

    def init_service(self, args):
        # Initialize the LLM service with the provided arguments
        print(f"Initializing LLM service...")
        self.llm = RAGAccLLM(args)

    def serve_request(self, data):
        result = super().serve_request(data)
        if result is not None:
            return result
        # Process the request and return the result
        if data.type == LLM_GENERATE_SIM_REQUEST:
            output = self.llm.llm_sim_generate_batch(
                data.args['batch_size'],
                data.args['input_lens'],
                data.args['output_lens'],
            )
            return Reply(status=REPLY_STATUS_OK, data={'output': output})
        raise ValueError(f"Unknown request type: {data.type}")


class RagService(Service):
    def __init__(self, args, port: int, byte_mode=False, stand_alone=False, numa_node=None):
        super().__init__(args, port, byte_mode, stand_alone, numa_node)
        self.ragacc = None
        self.args = None
        self.evaluation_func = None

    def init_service(self, args):
        # Initialize the RAGAcc service with the provided arguments
        print(f"Initializing RAGAcc service...")
        from .ragacc import RAGAcc
        self.ragacc = RAGAcc(args)
        self.args = args
        from .pipeline import rag_pipeline_evaluation
        self.evaluation_func = rag_pipeline_evaluation

    def serve_request(self, data):
        result = super().serve_request(data)
        if result is not None:
            return result
        # Process the request and return the result
        if data.type == RAG_PIPELINE_EVALUATION_REQUEST:
            result = self.evaluation_func(
                self.ragacc, data.args['pipeline'], self.args,
                data.args['data'], data.args['use_cluster_prefetch'],
                data.args['prefetch_budget'],
            )
            return Reply(status=REPLY_STATUS_OK, data={'result': result})
        elif data.type == RAG_WARM_UP_REQUEST:
            self.ragacc.warm_up_llm(
                data.args['warm_up'],
                data.args['prefetch_query'],
                data.args['prefetch_budget'],
            )
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RAG_TXT_TO_EMB_REQUEST:
            emb = self.ragacc.txt_to_emb(data.args['txt'])
            return Reply(status=REPLY_STATUS_OK, data={'emb': emb})
        elif data.type == RAG_CLEAR_ALL_PREFETCH_DATA_REQUEST:
            self.ragacc.clear_prefetch_data_on_all_gpus()
            return Reply(status=REPLY_STATUS_OK)
        elif data.type == RAG_UPDATE_NPROBE_REQUEST:
            self.args.nprobe = data.args['nprobe']
            return Reply(status=REPLY_STATUS_OK)
        raise ValueError(f"Unknown request type: {data.type}")


def namespace_to_args_list(namespace):
    args = []
    for key, value in vars(namespace).items():
        key_string = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(key_string)
        elif isinstance(value, list):
            for v in value:
                args.append(key_string)
                args.append(str(v))
        elif value is not None:
            args.append(key_string)
            args.append(str(value))
    return args

def add_env(new_env: dict) -> dict:
    """
    Add new environment variables to the current process.
    """
    env = os.environ.copy()
    for key, value in new_env.items():
        if isinstance(value, str):
            env[key] = value
        elif isinstance(value, (int, float)):
            env[key] = str(value)
        else:
            raise ValueError(f"Unsupported type for environment variable {key}: {type(value)}")
    return env

async def wait_service_initialization(address: str):
    await async_send_recv(
        address, Request(type=TEST_START_REQUEST, args={}), byte_mode=False,
    )


def construct_rag_service_cmd(config: dict, args):
    """Constructs the command to run the RAG service with the provided
    configuration and arguments.
    """
    numa_cmd = [
        'numactl', '-m', str(config['numa_node']),
        '-N', str(config['numa_node']), '--',
    ] if 'numa_node' in config else []
    return numa_cmd + [
        'python3', '-m', 'ragacc.rag_service',
        '--service-port', str(config['service_port']),
        '--retrieval-port', str(config['retrieval_port']),
        '--llm-port', str(config['llm_port']),
        '--retrieval-gpu-id', str(config['retrieval_gpu_id']),
        '--llm-gpu-id', str(config['llm_gpu_id']),
        '--nccl-port', str(config['nccl_port']),
    ] + [str(x) for x in namespace_to_args_list(args)]


def start_and_register_all_services(config, args):
    """
    Start all services based on the provided configuration and register
    them in the service manager. When starting the RAG service, it will
    be put to the same NUMA node as the LLM service because RAG service
    will have the embedding model.

    In this function, we only need to start the RAG service, which will
    start the LLM and retrieval services as subprocesses. However, the
    registration will include all three services (RAG, LLM, and Retrieval).

    Args:
        config (list[dict]): List of service configurations.
        args: Arguments for the services.
    """
    global service_manager

    # Start RAG services with Popen
    async def start_rag_service(i, cfg):
        time.sleep(20 * i)  # stagger the startups
        subprocess.Popen(
            construct_rag_service_cmd(cfg, args),
            env=add_env({'CUDA_VISIBLE_DEVICES': cfg['llm_gpu_id']}),
        )
        await wait_service_initialization(
            f"tcp://localhost:{cfg['service_port']}"
        )
    async def start_all_rag_services():
        await asyncio.gather(*[
            start_rag_service(i, cfg) for i, cfg in enumerate(config)
        ])
    asyncio.run(start_all_rag_services())

    # register services in the service manager
    for i, cfg in enumerate(config):
        # RAG service
        rag_service = ServiceInfo(
            addr=f"tcp://localhost:{cfg['service_port']}",
            port=cfg['service_port'],
        )
        service_manager.register_service('rag', i, rag_service)

        # Retrieval service
        retrieval_service = ServiceInfo(
            addr=f"tcp://localhost:{cfg['retrieval_port']}",
            port=cfg['retrieval_port'],
        )
        service_manager.register_service('retrieval', i, retrieval_service)

        # LLM service
        llm_service = ServiceInfo(
            addr=f"tcp://localhost:{cfg['llm_port']}",
            port=cfg['llm_port'],
        )
        service_manager.register_service('llm', i, llm_service)


service_manager = ServiceManager()
