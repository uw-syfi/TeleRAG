# Figure out what memory copy is needed for doing Cluster Prefetching

import time
import gc
import os
import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Tuple

from .const import (INIT_HOTNESS, HOTNESS_INCREASE, HOTNESS_DECAY,
                    CACHE_FRACTION_DEFAULT, SMALL_NUMBER)
from .faiss_utils import get_invlist


class RAGAccIndex:
    def __init__(self, args, faiss_index=None):
        self.device = torch.device(f"cuda:{args.gpu_id}")
        self.index_type = args.index_type
        self.disable_prefetch = args.disable_prefetch
        self.disable_gpu_sort = args.disable_gpu_sort
        self.disable_llm = args.disable_llm
        self.max_cpu_threads = args.max_cpu_threads

        self.index_load_dir = args.index_load_dir
        self.index_key = args.index_key
        self.faiss_index = None

        self.prefetch_clusters = torch.empty(0, dtype=torch.int64, device=self.device)
        self.prefetch_set = set()
        self.n_prefetch = 0
        self.cache_fraction = CACHE_FRACTION_DEFAULT

        if self.index_type == "faiss":
            self.disable_prefetch = True

        # pre-allocate virtual memory to hold prefetch data
        self.embed_dim = args.embed_dim
        size_in_gb = args.vm_size * 1024 * 1024 * 1024
        # convert vm_size to the number of entries in the prefetch memory pool
        self.vm_size = int(size_in_gb/(4 * self.embed_dim))

        if not args.disable_retrieval:
            if self.index_type == "faiss":
                if faiss_index is not None:
                    self.faiss_index = faiss_index
                else:
                    self.faiss_index = self.load_faiss_index(
                        f"{self.index_load_dir}/{self.index_key}.index",
                        use_faiss_gpu = args.use_faiss_gpu
                    )
                self.construct_invlist(self.faiss_index)

            elif self.index_type == "ragacc":
                if args.from_faiss:
                    if faiss_index is None:
                        faiss_index = self.load_faiss_index(f"{self.index_load_dir}/{self.index_key}.index")
                    self.construct_invlist(faiss_index)
                    if args.save_invlist_data:
                        self.save_invlist_data(self.index_load_dir, self.index_key)
                else:
                    self.load_invlist_data(self.index_load_dir, self.index_key)

            else:
                raise NotImplementedError
            self.centroids = self.centroids.to(self.device)
            self.pin_flat_invlist_data()

        if (not self.disable_prefetch) and (not args.disable_retrieval):
            self.allocate_gpu_buffers()

        # Deprecated: LLM model loading is not needed for index
        # print("Loading LLM model")
        # llm_args = IndexArgs.get_sglang_args(args)
        # self.model_runner, self.tokenizer = load_model(llm_args, 0, args.nccl_port)

        if not args.cpu_only:
            self.index_stream = torch.cuda.Stream()
            self.miss_stream = torch.cuda.Stream()

        # For multi GPU simulation
        self.sim_multi_gpu = args.sim_multi_gpu
        if self.sim_multi_gpu:
            self.num_gpu = args.num_gpu
            self.cached_clusters = [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(self.num_gpu)]
            self.hotness = [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(self.num_gpu)]
            self.current_gpu = 0
        else:
            # To avoid similar but slightly different code in different cases,
            # we still use a list to store the cached clusters and hotness.
            # The GPU ID will always be 0 in this case.
            self.num_gpu = 1
            self.cached_clusters = [torch.empty(0, dtype=torch.int64, device=self.device),]
            self.hotness = [torch.empty(0, dtype=torch.int64, device=self.device),]
            self.current_gpu = 0

        self.prefetch_runner = ThreadPoolExecutor(max_workers=128)

    def change_cache_fraction(self, new_fraction):
        """
        Change the cache fraction for prefetching.
        """
        assert 0 < new_fraction <= 1, "Cache fraction must be in (0, 1]."
        self.cache_fraction = new_fraction

    def change_num_gpu(self, num_gpu):
        """
        Change the number of GPUs for simulation.
        """
        if not self.sim_multi_gpu: return
        assert num_gpu > 0, "Number of GPUs must be greater than 0."
        self.clear_prefetch_data()
        if num_gpu == self.num_gpu: return
        self.num_gpu = num_gpu
        self.cached_clusters = [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(self.num_gpu)]
        self.hotness = [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(self.num_gpu)]
        self.current_gpu = 0

    def switch_gpu(self, gpu_id, update_cache_record=False):
        """
        Switch the GPU for prefetching. This function is for simulation
        of multiple GPUs only!

        The function will clear the prefetch data and load the cached
        clusters on the new GPU id. If update_cache_record is True,
        it will store the prefetch clusters for the current GPU in the
        cache record.
        """
        if not self.sim_multi_gpu: return
        assert gpu_id < self.num_gpu, "GPU ID exceeds the number of GPUs."
        if update_cache_record:
            # store the prefetch clusters for the current GPU
            self.cached_clusters[self.current_gpu] = self.prefetch_clusters
        # clear the prefetch data
        self.clear_prefetch_data(keep_cache=True)
        # load the prefetch clusters for the new GPU
        if self.cached_clusters[gpu_id].shape[0] == 0:
            # if the cached clusters are empty, do not prefetch
            self.prefetch_clusters = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            self.prefetch_with_cluster_list(self.cached_clusters[gpu_id])
        self.current_gpu = gpu_id

    def resize_cache_and_clear_for_next(self):
        """
        Resize the cache size to the target size.
        """
        # resize the cache size to the target size
        assert len(self.hotness[self.current_gpu]) == len(self.prefetch_clusters)
        if self.cache_fraction < SMALL_NUMBER:
            # if the cache fraction is 0, clear the cache
            self.clear_prefetch_data()
            return
        hotness_order = torch.argsort(self.hotness[self.current_gpu], descending=False)
        num_clusters = len(self.prefetch_clusters)

        # Calculate the total size occupied
        total_size = 0
        for i in range(num_clusters):
            cluster_id = self.prefetch_clusters[i]
            start_end = self.cluster_meta[cluster_id]
            n_data = start_end[1] - start_end[0]
            total_size += n_data

        # Start removing the least hot clusters
        target_size = int(self.vm_size * self.cache_fraction)
        mask = torch.full((num_clusters,), True, device=self.device)
        for i in range(num_clusters):
            if total_size <= target_size:
                break
            cluster_id = self.prefetch_clusters[hotness_order[i]]
            start_end = self.cluster_meta[cluster_id]
            n_data = start_end[1] - start_end[0]
            total_size -= n_data
            self.n_prefetch -= n_data
            mask[hotness_order[i]] = False
        # Remove the least hot clusters
        self.prefetch_clusters = self.prefetch_clusters[mask]
        self.hotness[self.current_gpu] = self.hotness[self.current_gpu][mask]
        self.cached_clusters[self.current_gpu] = self.prefetch_clusters
        self.prefetch_set = set(self.prefetch_clusters.tolist())
        # clear the prefetch data
        self.clear_prefetch_data(keep_cache=True)
        if self.cached_clusters[self.current_gpu].shape[0] != 0:
            self.prefetch_with_cluster_list(self.cached_clusters[self.current_gpu])


    def load_faiss_index(self, index_path, use_faiss_gpu=False):
        print("Loading Faiss Index")
        tick = time.time()
        faiss_index = faiss.read_index(index_path)
        print("Faiss Index loaded in: ", time.time() - tick)
        
        if use_faiss_gpu:
            print("Loading Faiss Index from CPU to GPU")
            tick = time.time()
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            print("Faiss Index loaded to GPU in: ", time.time() - tick)

        return faiss_index

    def find_clusters_faiss(self, emb, nprobe):
        _, Iq = self.faiss_index.quantizer.search(emb, nprobe)
        return Iq

    def set_nprobe_faiss(self, nprobe):
        self.faiss_index.nprobe = nprobe
    
    def allocate_gpu_buffers(self):
        print("Pre-allocating memory on GPU for Cluster Prefetching")
        tick = time.time()
        self.prefetch_ids_gpu = torch.empty(self.vm_size, dtype=int, device=self.device)
        self.prefetch_codes_gpu = torch.empty(self.vm_size, self.embed_dim, dtype=torch.float32, device=self.device)
        tock = time.time()
        print("Pre-allocating memory on GPU took (s): ", tock - tick)
    
    def construct_invlist(self, faiss_index):
        print("Constructing Invlist")
        faiss_invlist = faiss_index.invlists

        self.centroids = faiss_index.quantizer.reconstruct_n(0, faiss_index.nlist)
        self.centroids = torch.from_numpy(self.centroids)

        self.cluster_meta = []
        self.list_ids_tab = []
        self.list_codes_tab = []
        tick = time.time()
        accumulation = 0
        for i in range(faiss_invlist.nlist):
            list_ids, list_codes = get_invlist(faiss_invlist, i)
            list_ids = torch.from_numpy(list_ids)
            list_codes = torch.from_numpy(list_codes.view(np.uint32).view(np.float32))

            self.list_ids_tab.append(list_ids)
            self.list_codes_tab.append(list_codes)
            self.cluster_meta.append((accumulation, accumulation + list_ids.shape[-1]))

            accumulation += list_ids.shape[-1]

        self.list_ids_tab = torch.cat(self.list_ids_tab)
        self.list_codes_tab = torch.vstack(self.list_codes_tab)
        tock = time.time()
        print("Constructing Invlist took (s): ", tock - tick)

    def pin_flat_invlist_data(self):
        print("Pinning Invlist data")
        tick = time.time()
        self.list_ids_tab = self.list_ids_tab.pin_memory()
        self.list_codes_tab = self.list_codes_tab.pin_memory()
        tock = time.time()
        print("Pinning Invlist data took (s): ", tock - tick)

    def save_invlist_data(self, save_dir, index_key):
        print("Saving Invlist data")

        tick = time.time()
        full_path_cens = f'{save_dir}/{index_key}.centroids.pt'
        if not os.path.exists(full_path_cens):
            torch.save(self.centroids.cpu(), full_path_cens)

        full_path_ids = f'{save_dir}/{index_key}.list_ids_tab.pt'
        if not os.path.exists(full_path_ids):
            torch.save(self.list_ids_tab, full_path_ids)

        full_path_codes = f'{save_dir}/{index_key}.list_codes_tab.pt'
        if not os.path.exists(full_path_codes):
            torch.save(self.list_codes_tab, full_path_codes)

        full_path_meta = f'{save_dir}/{index_key}.cluster_meta.pt'
        if not os.path.exists(full_path_meta):
            torch.save(self.cluster_meta, full_path_meta)
        tock = time.time()
        print("Saving Invlist data took (s): ", tock - tick)

    def load_invlist_data(self, load_dir, index_key):
        print("Loading Invlist data from disk")
        assert os.path.exists(f"{load_dir}/{index_key}.centroids.pt")
        assert os.path.exists(f"{load_dir}/{index_key}.list_ids_tab.pt")
        assert os.path.exists(f"{load_dir}/{index_key}.list_codes_tab.pt")
        assert os.path.exists(f"{load_dir}/{index_key}.cluster_meta.pt")

        tick = time.time()
        self.centroids = torch.load(f"{load_dir}/{index_key}.centroids.pt", weights_only=True)
        self.list_ids_tab = torch.load(f"{load_dir}/{index_key}.list_ids_tab.pt", weights_only=True)
        self.list_codes_tab = torch.load(f"{load_dir}/{index_key}.list_codes_tab.pt", weights_only=True)
        self.cluster_meta = torch.load(f"{load_dir}/{index_key}.cluster_meta.pt", weights_only=True)
        tock = time.time()
        print("Loading Invlist data took (s): ", tock - tick)

    def find_clusters(self, emb, nprobe):
        x = torch.matmul(emb, self.centroids.T)
        ids = torch.argsort(-x)[:,:nprobe]
        return ids

    def search_faiss(self, emb, topk, nprobe=32):
        assert self.faiss_index is not None, "Faiss index is not loaded."
        self.set_nprobe_faiss(nprobe)
        emb = emb.cpu().numpy()
        # Prepare lists to collect results
        D_list = []
        I_list = []

        for i in range(len(emb)):
            # Slice the query to shape (1, d) using i:i+1
            # If you use emb[i], it becomes (d,), which FAISS will reject.
            q = emb[i:i+1]
            
            # Search one by one
            D_i, I_i = self.faiss_index.search(q, topk)
            
            # Collect results
            D_list.append(D_i)
            I_list.append(I_i)

        # Stack them back into the standard (N, topk) shape
        D = np.vstack(D_list)
        I = np.vstack(I_list)
        return D, I
        # return D[0], I[0]

    def copy_prefetch_data_to_gpu(self, cpu_range, gpu_range):
        self.prefetch_ids_gpu[gpu_range[0]:gpu_range[1]].copy_(
            self.list_ids_tab[cpu_range[0]:cpu_range[1]], non_blocking=True
        )
        self.prefetch_codes_gpu[gpu_range[0]:gpu_range[1]].copy_(
            self.list_codes_tab[cpu_range[0]:cpu_range[1]], non_blocking=True
        )

    def clear_prefetch_data(self, keep_cache=False):
        """clear the prefetch data on GPU

        args:
            keep_cache (bool): (INTERNAL ONLY!!) whether to keep the cached
                clusters for multi-GPU simulation. This will result in an
                intermediate state as the prefetch_clusters are not consistent
                with the cached_clusters. Only use this when switching
                between GPUs in multi-GPU simulation.
        """
        torch.cuda.synchronize()
        self.prefetch_set.clear()
        gc.collect()

        self.prefetch_clusters = torch.empty(0, dtype=torch.int64, device=self.device)
        if not keep_cache:
            self.cached_clusters[self.current_gpu] = torch.empty(0, dtype=torch.int64, device=self.device)
            self.hotness[self.current_gpu] = torch.empty(0, dtype=torch.int64, device=self.device)
        self.n_prefetch = 0
    
    def simulate_prefetch(self, emb, budget=None, nprobe=1024):
        assert not self.disable_prefetch, "Prefetching is disabled."
        assert emb.shape[0] == 1, "Only single batch prefetching is supported."

        if budget is None:
            prefetch_budget = self.vm_size
        else:
            budget_in_gb = budget * 1024 * 1024 * 1024
            prefetch_budget = int(budget_in_gb/(4 * self.embed_dim))

        with torch.cuda.stream(self.index_stream):
            Iq = self.find_clusters(emb, nprobe=nprobe)

            mask = torch.isin(Iq[0], self.prefetch_clusters)
            miss_clusters = Iq[0][~mask]

            _n_prefetch = 0
            _prefetch_clusters = []
            for c_id in miss_clusters:
                start_end = self.cluster_meta[int(c_id)]
                n_data = start_end[1] - start_end[0]

                if self.n_prefetch + n_data > self.vm_size:
                    break
            
                if _n_prefetch + n_data > prefetch_budget:
                    break

                _prefetch_clusters.append(c_id)

                self.n_prefetch += n_data
                _n_prefetch += n_data
            
            _prefetch_clusters = torch.tensor(_prefetch_clusters, device=self.device)
            self.prefetch_clusters = torch.cat([self.prefetch_clusters, _prefetch_clusters]) 

        # print(f"Number of prefetch clusters: {len(self.prefetch_clusters)}")
        return self.n_prefetch, len(self.prefetch_clusters)

    def get_cluster_hit_rate(
            self, emb, nprobe=32,
        ):
        Iq = self.find_clusters(emb, nprobe=nprobe)

        batch = emb.shape[0]

        if self.prefetch_clusters is not None:
            mask = torch.isin(Iq, self.prefetch_clusters)

            miss_clusters = []
            for i in range(batch):
                miss_clusters.append(Iq[i][~mask[i]])

            hit_rate_arr = []
            for i in range(batch):
                hit_rate = 1 - len(miss_clusters[i])/len(Iq[i])
                hit_rate_arr.append(hit_rate)
            hit_rate = np.mean(hit_rate_arr)
        else:
            hit_rate = 0.0

        return hit_rate

    def prefetch(self, emb, budget=None, nprobe=1024, sync=False):
        """This function will be removed and replaced by prefetch_batch() in the future."""
        return self.prefetch_batch(self, [emb], budget, nprobe, sync)
        # assert not self.disable_prefetch, "Prefetching is disabled."
        # assert emb.shape[0] == 1, "Only single batch prefetching is supported."
        #
        # if budget is None:
        #     prefetch_budget = self.vm_size
        # else:
        #     budget_in_gb = budget * 1024 * 1024 * 1024
        #     prefetch_budget = int(budget_in_gb/(4 * self.embed_dim))
        #
        # with torch.cuda.stream(self.index_stream):
        #     Iq = self.find_clusters(emb, nprobe=nprobe)
        #
        #     mask = torch.isin(Iq[0], self.prefetch_clusters)
        #     miss_clusters = Iq[0][~mask]
        #
        #     cpu_start_end = []
        #     gpu_start_end = []
        #     _n_prefetch = 0
        #     _prefetch_clusters = []
        #     for c_id in miss_clusters:
        #         start_end = self.cluster_meta[int(c_id)]
        #         n_data = start_end[1] - start_end[0]
        #
        #         if self.n_prefetch + n_data > self.vm_size:
        #             break
        #
        #         if _n_prefetch + n_data > prefetch_budget:
        #             break
        #
        #         _prefetch_clusters.append(c_id)
        #         cpu_start_end.append(start_end)
        #         gpu_start_end.append((self.n_prefetch, self.n_prefetch+n_data))
        #
        #         self.n_prefetch += n_data
        #         _n_prefetch += n_data
        #
        #     _prefetch_clusters = torch.tensor(_prefetch_clusters, device=self.device)
        #     self.prefetch_clusters = torch.cat([self.prefetch_clusters, _prefetch_clusters])
        #
        #     # Launching non-blocking memory copy operations
        #     # NEED MANUAL SYNCHRONIZATION before performing search
        #     for i in range(len(_prefetch_clusters)):
        #         cpu_range = cpu_start_end[i]
        #         gpu_range = gpu_start_end[i]
        #         self.copy_prefetch_data_to_gpu(cpu_range, gpu_range)
        #
        # if sync:
        #     torch.cuda.synchronize()
        #
        # return self.n_prefetch, len(self.prefetch_clusters)

    def prefetch_batch(self, emb, budget=None, nprobe=1024, sync=False):
        assert not self.disable_prefetch, "Prefetching is disabled."

        if budget is None:
            prefetch_budget = self.vm_size
        else:
            budget_in_gb = budget * 1024 * 1024 * 1024
            prefetch_budget = int(budget_in_gb/(4 * self.embed_dim))

        with torch.cuda.stream(self.index_stream):
            Iq = self.find_clusters(emb, nprobe=nprobe)
            batch = emb.shape[0]

            prefetch_list = [Iq[i].tolist() for i in range(batch)]
            cpu_start_end = []
            gpu_start_end = []
            _n_prefetch = 0
            _prefetch_clusters = []
            hit_clusters = []
            for i in range(nprobe):
                for j in range(batch):
                    element = prefetch_list[j][i]
                    if element not in self.prefetch_set:

                        start_end = self.cluster_meta[int(element)]
                        n_data = start_end[1] - start_end[0]

                        if self.n_prefetch + n_data > self.vm_size:
                            break

                        if _n_prefetch + n_data > prefetch_budget:
                            break

                        self.prefetch_set.add(element)

                        _prefetch_clusters.append(element)
                        cpu_start_end.append(start_end)
                        gpu_start_end.append((self.n_prefetch, self.n_prefetch+n_data))

                        self.n_prefetch += n_data
                        _n_prefetch += n_data
                    else:
                        hit_clusters.append(element)

            _prefetch_clusters = torch.tensor(_prefetch_clusters, device=self.device)
            has_prefetch = len(_prefetch_clusters) != 0
            all_miss = len(hit_clusters) == 0
            init_with_no_clusters = len(self.prefetch_clusters) == 0
            hit_clusters = torch.tensor(hit_clusters, device=self.device)

            # Hotness & cache recomputed, and prefetched clusters updated
            self.hotness[self.current_gpu] //= HOTNESS_DECAY
            if not all_miss and not init_with_no_clusters:
                hit_cluster_mask = torch.isin(self.prefetch_clusters, hit_clusters)
                self.hotness[self.current_gpu][hit_cluster_mask] += HOTNESS_INCREASE
            if has_prefetch:
                hotness_to_add = torch.full((len(_prefetch_clusters),), INIT_HOTNESS, device=self.device)
                self.hotness[self.current_gpu] = torch.cat([self.hotness[self.current_gpu], hotness_to_add])
                self.prefetch_clusters = torch.cat([self.prefetch_clusters, _prefetch_clusters])
            self.cached_clusters[self.current_gpu] = self.prefetch_clusters

            # Launching non-blocking memory copy operations
            # NEED MANUAL SYNCHRONIZATION before performing search
            for i in range(len(_prefetch_clusters)):
                cpu_range = cpu_start_end[i]
                gpu_range = gpu_start_end[i]
                self.copy_prefetch_data_to_gpu(cpu_range, gpu_range)

        if sync:
            torch.cuda.synchronize()

        return self.n_prefetch, len(self.prefetch_clusters)

    def prefetch_with_cluster_list(self, cluster_list):
        """
        This function is temporarily for bringing back the cached clusters
        only! If anyone wants to use this function for other usage, check
        the difference between this function and prefetch_batch().
        """
        assert not self.disable_prefetch, "Prefetching is disabled."
        assert len(cluster_list) > 0, "Cluster list is empty."

        with torch.cuda.stream(self.index_stream):
            _prefetch_clusters = cluster_list
            self.prefetch_clusters = torch.cat([self.prefetch_clusters, _prefetch_clusters])

            cpu_start_end = []
            gpu_start_end = []
            _n_prefetch = 0
            for i in range(cluster_list.shape[0]):
                c_id = cluster_list[i].item()
                start_end = self.cluster_meta[int(c_id)]
                n_data = start_end[1] - start_end[0]

                if self.n_prefetch + n_data > self.vm_size:
                    break

                cpu_start_end.append(start_end)
                gpu_start_end.append((self.n_prefetch, self.n_prefetch+n_data))

                self.n_prefetch += n_data
                _n_prefetch += n_data

            # Launching non-blocking memory copy operations
            # NEED MANUAL SYNCHRONIZATION before performing search
            for i in range(len(cluster_list)):
                cpu_range = cpu_start_end[i]
                gpu_range = gpu_start_end[i]
                self.copy_prefetch_data_to_gpu(cpu_range, gpu_range)
        self.prefetch_set.update(cluster_list.tolist())
        return self.n_prefetch, len(self.prefetch_clusters)

    def search_prefetch_gpu(self, emb_gpu, topk):
        # dists_all = torch.matmul(self.prefetch_codes_gpu[:self.n_prefetch], emb_gpu.T).squeeze(1)
        dists_all = torch.matmul(self.prefetch_codes_gpu[:self.n_prefetch], emb_gpu.T)
        sort_ids = torch.argsort(-dists_all.T)[:,:topk]

        dists, ids = [], []
        # TODO: check if possible to use torch.gather or broadcasting 
        for i in range(sort_ids.shape[0]):
            dists.append(dists_all[sort_ids[i],i])
            ids.append(self.prefetch_ids_gpu[sort_ids[i]])

        return torch.stack(dists, axis=0), torch.stack(ids, axis=0)

    def compute_dist_for_single_cluster_cpu(self, c_id, emb):
        id_range = self.cluster_meta[c_id]
        dists = torch.matmul(self.list_codes_tab[id_range[0]:id_range[1]], emb.T).squeeze(1)
        ids = self.list_ids_tab[id_range[0]:id_range[1]]

        return dists, ids

    # search the miss clusters on CPU
    def search_miss_cpu(self, miss_clusters, emb, topk):
        if len(miss_clusters) == 0:
            return None, None

        emb = emb.cpu()

        threads = []
        with ThreadPoolExecutor(max_workers=self.max_cpu_threads) as executor:
            for c_id in miss_clusters:
                threads.append(executor.submit(self.compute_dist_for_single_cluster_cpu, c_id, emb))
        wait(threads)

        dists_all = torch.cat([t.result()[0] for t in threads])
        ids_all = torch.cat([t.result()[1] for t in threads])

        with torch.cuda.stream(self.miss_stream):
            # move data to GPU for sorting if not disabled
            if not self.disable_gpu_sort:
                dists_all = dists_all.to(self.device)
                ids_all = ids_all.to(self.device)

            sort_ids = torch.argsort(-dists_all)[:topk]
            dists = dists_all[sort_ids]
            ids = ids_all[sort_ids]

        return dists, ids

    def merge_search_results(self, d_gpu, i_gpu, d_cpu, i_cpu, topk):
        batch = d_gpu.shape[0]

        dists, ids = [], []
        for i in range(batch):
            if i_cpu[i] == None:
                d_all = d_gpu[i]
                i_all = i_gpu[i]
            else:
                d_all = torch.cat([d_gpu[i], d_cpu[i]])
                i_all = torch.cat([i_gpu[i], i_cpu[i]])

            sort_ids = torch.argsort(-d_all)[:topk]

            dists.append(d_all[sort_ids])
            ids.append(i_all[sort_ids])

        return torch.stack(dists, axis=0), torch.stack(ids, axis=0)

    def search_ragacc_gpu_only(
            self, emb, topk,
            runtime_fetch=False, fetch_emb=None, fetch_nprobe=None,
        ):
        if runtime_fetch:
            # preform data fetching to GPU at runtime
            assert fetch_emb is not None, "fetch_emb shall not be None."
            assert fetch_nprobe is not None, "fetch_nprobe shall not be None."
            self.clear_prefetch_data()
            self.prefetch(fetch_emb, nprobe=fetch_nprobe)
            torch.cuda.synchronize()

        assert self.prefetch_clusters is not None, "No prefetch data on GPU, call prefetch() first for GPU-only search."
        # print("Perform Prefetching-only (GPU-only) search")

        D, I = self.search_prefetch_gpu(emb, topk)

        return D.cpu().tolist(), I.cpu().tolist()

    def search_ragacc_cpu_only(self, emb, topk, nprobe):
        Iq = self.find_clusters(emb, nprobe=nprobe)
        batch = emb.shape[0]

        threads = []
        with ThreadPoolExecutor() as executor:
            for i in range(batch):
                threads.append(executor.submit(self.search_miss_cpu, Iq[i], emb[i,:].unsqueeze(0), topk))
        wait(threads)

        D, I = [], []
        for i in range(batch):
            d_cpu, i_cpu = threads[i].result()
            D.append(d_cpu)
            I.append(i_cpu)
        D = torch.stack(D, axis=0)
        I = torch.stack(I, axis=0)

        return D.tolist(), I.tolist()

    def search_ragacc(
            self, emb, topk, nprobe=32,
            runtime_fetch=False, fetch_emb=None, fetch_nprobe=None,
        ):
        if runtime_fetch:
            # preform data fetching to GPU at runtime
            assert fetch_emb is not None, "fetch_emb shall not be None."
            assert fetch_nprobe is not None, "fetch_nprobe shall not be None."
            self.prefetch(fetch_emb, nprobe=fetch_nprobe)

        # print("Search with Regular RAGACC Search")
        Iq = self.find_clusters(emb, nprobe=nprobe)

        batch = emb.shape[0]

        if self.prefetch_clusters is not None:
            mask = torch.isin(Iq, self.prefetch_clusters)

            miss_clusters = []
            # TODO: check if possible to parallelize this
            for i in range(batch):
                miss_clusters.append(Iq[i][~mask[i]])

            threads = []
            with ThreadPoolExecutor() as executor:
                threads.append(executor.submit(self.search_prefetch_gpu, emb, topk))

                for i in range(batch):
                    threads.append(executor.submit(self.search_miss_cpu, miss_clusters[i], emb[i,:].unsqueeze(0), topk))
            wait(threads)

            Dgpu, Igpu = threads[0].result()
            Dcpu, Icpu = [], []
            for i in range(1,batch+1):
                d_cpu, i_cpu = threads[i].result()
                Dcpu.append(d_cpu)
                Icpu.append(i_cpu)

            D, I = self.merge_search_results(Dgpu, Igpu, Dcpu, Icpu, topk)
        else:
            threads = []
            with ThreadPoolExecutor() as executor:
                for i in range(batch):
                    threads.append(executor.submit(self.search_miss_cpu, Iq[i], emb[i,:].unsqueeze(0), topk))
            wait(threads)

            D, I = [], []
            for i in range(batch):
                d_cpu, i_cpu = threads[i].result()
                D.append(d_cpu)
                I.append(i_cpu)
            D = torch.stack(D, axis=0)
            I = torch.stack(I, axis=0)

        return D.cpu().tolist(), I.cpu().tolist()

    def search(
            self, emb, topk, nprobe=32, 
            gpu_only_search=False, cpu_only_search=False,
            runtime_fetch=False, fetch_emb=None, fetch_nprobe=None, 
        ):
        if runtime_fetch:
            assert self.index_type == "ragacc", "Runtime fetch is only supported for RAGAcc index."

        if self.index_type == "faiss":
            return self.search_faiss(emb, topk, nprobe)
        elif self.index_type == "ragacc":
            # if self.gpu_only_search:
            if gpu_only_search:
                return self.search_ragacc_gpu_only(
                    emb, topk, runtime_fetch, fetch_emb, fetch_nprobe
                )
            elif cpu_only_search:
                return self.search_ragacc_cpu_only(emb, topk, nprobe)
            else:
                return self.search_ragacc(
                    emb, topk, nprobe, 
                    runtime_fetch, fetch_emb, fetch_nprobe,
                )
        else:
            raise NotImplementedError

    @torch.inference_mode()
    def get_cache_clusters_overlap(self, emb, nprobe: int) -> Tuple[int, int]:
        """
        Get the number of cache clusters that overlap with the clusters
        corresponding to the input embeddings.
        """
        return self.get_cache_clusters_overlap_sim(emb, nprobe, self.current_gpu)

    @torch.inference_mode()
    def get_cache_clusters_overlap_sim(self, emb, nprobe: int, gpu_id: int) -> Tuple[int, int]:
        """
        Get the number of cache clusters that overlap with the clusters
        corresponding to the input embeddings.
        """
        emb = emb.to(self.device)

        Iq = self.find_clusters(emb, nprobe=nprobe)  # [B, nprobe]

        # Flatten and deduplicate the cluster indices
        clusters = torch.unique(Iq)

        # Compute and return the number of overlapping clusters
        return torch.isin(clusters, self.cached_clusters[gpu_id]).sum().item(), clusters.shape[0]

    def check_consistency(self):
        """
        Check the consistency between the prefetch clusters and the cached clusters.
        This function is for debugging only.
        """
        assert len(self.prefetch_clusters) == \
               len(self.cached_clusters[self.current_gpu]), \
            "Inconsistent number of clusters between prefetch and cache."
        assert len(self.hotness[self.current_gpu]) == \
               len(self.cached_clusters[self.current_gpu]), \
            "Inconsistent number of hotness between prefetch and cache."

    # @torch.inference_mode()
    # def llm_generate(
    #     self, prompts, output_len,
    #     # prefetch=False, prefetch_emb=None, prefetch_nprobe=4,
    #     prefetch=False, prefetch_emb=None, prefetch_budget=2,
    # ):
    #     # current version is for single batch
    #     assert not self.disable_llm
    #
    #     # Clear the pools.
    #     self.model_runner.req_to_token_pool.clear()
    #     self.model_runner.token_to_kv_pool.clear()
    #
    #     input_ids, reqs = prepare_llm_inputs(
    #         self.tokenizer, prompts, output_len
    #     )
    #
    #     if prefetch:
    #         assert prefetch_emb is not None, "prefetch emb shall not be None."
    #         self.prefetch_runner.submit(self.prefetch, prefetch_emb, prefetch_budget)
    #
    #     next_token_ids, _, batch = extend(reqs, self.model_runner)
    #
    #     output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    #     for i in range(output_len):
    #         next_token_ids, _ = decode(next_token_ids, batch, self.model_runner)
    #         for j in range(len(reqs)):
    #             output_ids[j].append(next_token_ids[j])
    #
    #     output_txts = [self.tokenizer.decode(output_ids[j]) for j in range(len(reqs))]
    #
    #     torch.cuda.synchronize()
    #     return output_txts
    #
    # @torch.inference_mode()
    # def llm_generate_sim_batch(
    #     self, batch_size, input_lens, output_lens,
    #     # prefetch=False, prefetch_emb=None, prefetch_nprobe=4,
    #     prefetch=False, prefetch_emb=None, prefetch_budget=2,
    #     sync=True
    # ):
    #     assert not self.disable_llm, "LLM is disabled."
    #
    #     reqs = prepare_synthetic_llm_inputs_batch(
    #         batch_size, input_lens, output_lens
    #     )
    #
    #     # Clear the pools.
    #     self.model_runner.req_to_token_pool.clear()
    #     self.model_runner.token_to_kv_pool.clear()
    #
    #     max_output_len = max(output_lens)
    #
    #     if prefetch:
    #         assert prefetch_emb is not None, "prefetch emb shall not be None."
    #         self.prefetch_runner.submit(self.prefetch_batch, prefetch_emb, prefetch_budget)
    #
    #     next_token_ids, _, batch = extend(reqs, self.model_runner)
    #     for i in range(max_output_len):
    #         next_token_ids, _ = decode(next_token_ids, batch, self.model_runner)
    #
    #     if sync:
    #         torch.cuda.synchronize()
    #
    # @torch.inference_mode()
    # def llm_generate_sim_batch_no_prefetch(
    #     self, batch_size, input_lens, output_lens,
    # ):
    #     assert not self.disable_llm, "LLM is disabled."
    #
    #     reqs = prepare_synthetic_llm_inputs_batch(
    #         batch_size, input_lens, output_lens
    #     )
    #
    #     # Clear the pools.
    #     self.model_runner.req_to_token_pool.clear()
    #     self.model_runner.token_to_kv_pool.clear()
    #
    #     max_output_len = max(output_lens)
    #
    #     next_token_ids, _, batch = extend(reqs, self.model_runner)
    #     for i in range(max_output_len):
    #         next_token_ids, _ = decode(next_token_ids, batch, self.model_runner)
