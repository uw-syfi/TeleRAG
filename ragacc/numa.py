from ctypes import *
from ctypes.util import find_library

__all__ = [
    'numa_run_on_node', 'numa_id_from_gpu_id', 'gpu_actual_id_from_gpu_id',
]

LIBNUMA = CDLL(find_library("numa"))

MAX_NUMANODES = LIBNUMA.numa_num_possible_nodes()
NR_CPUS = LIBNUMA.numa_num_possible_cpus()

def numa_run_on_node(node: int):
    op_res = LIBNUMA.numa_run_on_node(node)
    if op_res == -1:
        raise RuntimeError(f"Failed to run numa on node {node}")
    res = ",".join(list(map(str, [node])))
    c_string = bytes(res, "ascii")
    bitmask = LIBNUMA.numa_parse_nodestring(c_string)
    op_res = LIBNUMA.numa_set_membind(bitmask)
    if op_res == -1:
        raise RuntimeError(f"Failed to run numa on node {node}")


def numa_id_from_gpu_id(gpu_id: int) -> int:
    """
    Get the NUMA node ID from the GPU ID.
    """
    # Assuming a simple mapping where each GPU corresponds to a NUMA node
    numa_node = gpu_id % 2
    return numa_node

def gpu_actual_id_from_gpu_id(gpu_id: int) -> int:
    """
    Get the actual GPU ID from the given GPU ID.
    """
    # Assuming a simple mapping where each GPU corresponds to its ID
    return numa_id_from_gpu_id(gpu_id) * 2 + gpu_id // 2
