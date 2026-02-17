import numpy as np
import time
import torch


def run_index(index, args):
    emb = np.random.random((1, 768)).astype(np.float32)
    emb = torch.from_numpy(emb).to(index.device)

    index.prefetch(emb, args.nprobe)
    torch.cuda.synchronize()

    Ddev, Idev = index.search(emb, args.topk, args.nprobe)
    print(Idev)

def test_index_hit_rate(index, args):
    emb = np.random.random((1, 768)).astype(np.float32)
    emb = torch.from_numpy(emb).to(index.device)

    index.prefetch(emb, args.nprobe)
    torch.cuda.synchronize()

    Ddev, Idev, hit_rate = index.search(emb, args.topk, args.nprobe, return_hit_rate=True)
    print(Idev)
    print(hit_rate)

    noise = np.random.random((1, 768)).astype(np.float32)
    noise = torch.from_numpy(noise).to(index.device)
    emb_noise = emb + noise * 0.01

    print("Adding noise to emb for prefetch")
    index.prefetch(emb_noise, args.nprobe)
    torch.cuda.synchronize()

    Ddev, Idev, hit_rate = index.search(emb, args.topk, args.nprobe, return_hit_rate=True)
    print(Idev)
    print(hit_rate)

    print("Testing batch case")
    batch = 4
    emb = np.random.random((4, 768)).astype(np.float32)
    emb = torch.from_numpy(emb).to(index.device)
    emb_mean = emb.mean(dim=0, keepdim=True)

    index.prefetch(emb_mean, args.nprobe)
    torch.cuda.synchronize()
    Ddev, Idev, hit_rate = index.search(emb, args.topk, args.nprobe, return_hit_rate=True)
    print(Idev)
    print(hit_rate)

def bench_index(faiss_index, index, args, n_run=1):
    emb = np.random.random((1, 768)).astype(np.float32)

    faiss_index.nprobe = args.nprobe

    # warm up
    for _ in range(3):
        Dref, Iref = faiss_index.search(emb, args.topk)

    tick = time.time()
    for _ in range(n_run):
        Dref, Iref = faiss_index.search(emb, args.topk)
    faiss_t = time.time() - tick
    print(f"Faiss search time: {faiss_t/n_run:.3f}")
    # print(Iref)

    emb = torch.from_numpy(emb).to(index.device)

    # tick = time.time()
    # # results = index.prefetch(emb, args.nprobe)
    # with ThreadPoolExecutor() as executor:
    #     pf_thread = executor.submit(index.prefetch, emb, args.nprobe)
    # wait([pf_thread])
    # torch.cuda.synchronize()

    tock = time.time() - tick
    print(f"Prefetch time: {tock}")

    # warm up
    for _ in range(3):
        Ddev, Idev = index.search(emb, args.topk, args.nprobe, gpu_sort=False)

    tick = time.time()
    for _ in range(n_run):
        Ddev, Idev = index.search(emb, args.topk, args.nprobe, gpu_sort=False)
    ragacc_t = time.time() - tick
    print(f"Our search time: {ragacc_t/n_run:.3f}")
    # print(Idev)

    assert np.allclose(Iref[0], Idev), "Seach results are not equal"
    print("Search results are equal: PASS!")

def verify_index(faiss_index, index, args):
    emb = np.random.random((1, 768)).astype(np.float32)

    faiss_index.nprobe = args.nprobe
    tick = time.time()
    Dref, Iref = faiss_index.search(emb, args.topk)
    print(f"Faiss search time: {time.time() - tick}")
    print("Faiss results: ", Iref[0])

    emb = torch.from_numpy(emb).to(index.device)

    # tick = time.time()
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(index.prefetch, emb, args.nprobe)
    # torch.cuda.synchronize()
    # tock = time.time() - tick
    # print(f"Prefetch time: {tock}")

    tick = time.time()
    Ddev, Idev = index.search(emb, args.topk, args.nprobe)
    print(f"Our search time: {time.time() - tick}")
    print("Our results: ", Idev)

    assert np.allclose(Iref[0], Idev), "Seach results are not equal"
    print("Search results are equal: PASS!")

def test_index(index, args):
    warm_up = 2
    test_run = 5

    emb = np.random.random((1, 768)).astype(np.float32)

    index.set_nprobe(args.nprobe)
    Dref, Iref = index.search_faiss(emb, args.topk)

    tick = time.time()
    index.prefetch_clusters_cpu(emb, args.nprobe)
    find_clusters_t = time.time() - tick
    print(f"Find prefetch clusters time: {find_clusters_t}")
    Dcpu, Icpu = index.search_prefetch_cpu(emb, args.topk)
    assert np.allclose(Icpu, Iref[0])

    tick = time.time()
    # index.copy_prefetch_data_to_gpu()
    index.copy_prefetch_data_to_gpu_v2()
    torch.cuda.synchronize()
    copy_t = time.time() - tick
    print(f"Move data time: {copy_t}")
    emb_gpu = torch.from_numpy(emb).to(index.device)
    Dgpu, Igpu = index.search_prefetch_gpu(emb_gpu, args.topk)
    assert np.allclose(Igpu, Iref[0])

    for _ in range(warm_up):
        Dcpu, Icpu = index.search_prefetch_cpu(emb, args.topk)

    cpu_t = []
    for _ in range(test_run):
        tick = time.time()
        Dcpu, Icpu = index.search_prefetch_cpu(emb, args.topk)
        cpu_t.append(time.time() - tick)
    print(f"CPU search time: {np.mean(cpu_t)}")

    for _ in range(warm_up):
        Dgpu, Igpu = index.search_prefetch_gpu(emb_gpu, args.topk)

    gpu_t = []
    for _ in range(test_run):
        tick = time.time()
        Dcpu, Icpu = index.search_prefetch_gpu(emb_gpu, args.topk)
        torch.cuda.synchronize()
        gpu_t.append(time.time() - tick)
    print(f"GPU search time: {np.mean(gpu_t)}")