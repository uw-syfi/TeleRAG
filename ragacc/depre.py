def search_miss_cpu_v1(self, miss_clusters, emb, topk):
    miss_ids = []
    miss_codes = []
    tick = time.time()
    for c in miss_clusters:
        miss_ids.append(self.list_ids_tab[self.cluster_meta[c][0]:self.cluster_meta[c][1]])
        miss_codes.append(self.list_codes_tab[self.cluster_meta[c][0]:self.cluster_meta[c][1]])
    tock_0 = time.time()

    miss_ids = torch.cat(miss_ids)
    miss_codes = torch.vstack(miss_codes)
    tock_1 = time.time()
    
    dists_all = torch.matmul(miss_codes, emb.T).squeeze(1)
    tock_2 = time.time()

    sort_ids = torch.argsort(-dists_all)[:topk]

    dists = dists_all[sort_ids]
    ids = miss_ids[sort_ids]
    tock_3 = time.time()

    print(f"cpu search breakdown: {tock_0 - tick}, {tock_1 - tock_0}, {tock_2 - tock_1}, {tock_3 - tock_2}")

    return dists.tolist(), ids.tolist()

def search_miss_cpu_v2(self, miss_clusters, emb, topk):
    if len(miss_clusters) == 0:
        return None, None

    dist_arr = []
    ids_arr = []
    tick = time.time()
    for c_id in miss_clusters:
        id_range = self.cluster_meta[c_id]
        dist = torch.matmul(self.list_codes_tab[id_range[0]:id_range[1]], emb.T).squeeze(1)
        ids = self.list_ids_tab[id_range[0]:id_range[1]]

        dist_arr.append(dist)
        ids_arr.append(ids)
    tock_0 = time.time()

    dists_all = torch.cat(dist_arr)
    ids_all = torch.cat(ids_arr)
    tock_1 = time.time()

    sort_ids = torch.argsort(-dists_all)[:topk]
    dists = dists_all[sort_ids]
    ids = ids_all[sort_ids]
    tock_2 = time.time()

    print(f"cpu search breakdown: {tock_0 - tick}, {tock_1 - tock_0}, {tock_2 - tock_1},")

    return dists.tolist(), ids.tolist()
    
    # def merge_search_results(self, d_all, i_all, topk):
    # def merge_search_results(self, d_gpu, d_cpu, i_gpu, i_cpu, topk):
    #     d_all = torch.cat([d_gpu, d_cpu], axis=1)
    #     i_all = torch.cat([i_gpu, i_cpu], axis=1)

    #     sort_ids = torch.argsort(-d_all)[:, :topk]

    #     # dists = d_all[sort_ids]
    #     # ids = i_all[sort_ids]
    #     dists = [d_all[i, sort_ids[i]] for i in range(sort_ids.shape[0])]
    #     ids = [i_all[i, sort_ids[i]] for i in range(sort_ids.shape[0])]

    #     return torch.stack(dists, axis=0), torch.stack(ids, axis=0)

@nvtx.annotate(message="search", color="yellow")
def search(self, emb, topk, nprobe=32):
    if not self.disable_prefetch:
        Iq = self.find_clusters(emb, nprobe=nprobe)

        if self.prefetch_clusters is None:
            miss_clusters = Iq
        else:
            mask = torch.isin(Iq, self.prefetch_clusters)
            miss_clusters = Iq[~mask]

        print("# Miss clusters: ", miss_clusters.shape[0])

        with ThreadPoolExecutor() as executor:
            gpu_search = executor.submit(self.search_prefetch_gpu, emb, topk)
            cpu_search = executor.submit(self.search_miss_cpu, miss_clusters, emb, topk)
            threads = [gpu_search, cpu_search]
        wait(threads)

        Dgpu, Igpu = threads[0].result()
        Dcpu, Icpu = threads[1].result()
        if Icpu != None:
            D, I = self.merge_search_results([Dgpu, Dcpu], [Igpu, Icpu], topk)
        else:
            D, I = Dgpu, Igpu
    else:
        raise NotImplementedError
    return D.cpu().tolist(), I.cpu().tolist()
    
    @torch.inference_mode()
    def llm_generate_sim(
        self, batch_size, input_len, output_len,
        prefetch=False, prefetch_emb=None, prefetch_nprobe=4
    ):
        assert not self.disable_llm, "LLM is disabled."

        reqs = prepare_synthetic_llm_inputs(
            batch_size, input_len, output_len
        )

        # Clear the pools.
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool.clear()

        next_token_ids, _, batch = extend(reqs, self.model_runner)
        for i in range(output_len):
            if (i == 1) & (prefetch == True):
                assert prefetch_emb is not None, "prefetch emb shall not be None."
                self.prefetch_runner.submit(self.prefetch, prefetch_emb, prefetch_nprobe)
            next_token_ids, _ = decode(next_token_ids, batch, self.model_runner)
        torch.cuda.synchronize()
    
    @torch.inference_mode()
    def bench_llm(
            self, 
            llm_input: str,
            llm_output: str,
            prefetch_query: str = None,
            warm_up: int = 3,
            n_run: int = 5,
        ):
        """
        Only support single-batch LLM call with specified input and output length
        """
        input_len = self.count_token(llm_input)
        output_len = self.count_token(llm_output)
        # print(f"input_len: {input_len}, output_len: {output_len}")

        prefetch_emb = None
        prefetch = False if prefetch_query is None else True

        for _ in range(warm_up):
            self.index.llm_generate_sim(
                batch_size=1, input_len=30, output_len=10,
            )

        t_arr = []
        for _ in range(n_run):
            tick = time.time()
            if prefetch == True:
                prefetch_emb = self.txt_to_emb(prefetch_query)
            self.index.llm_generate_sim(
                batch_size=1, input_len=input_len, output_len=output_len,
                prefetch=prefetch, prefetch_emb=prefetch_emb, prefetch_nprobe=self.nprobe
            )
            t_arr.append(time.time() - tick)

        return np.mean(t_arr)