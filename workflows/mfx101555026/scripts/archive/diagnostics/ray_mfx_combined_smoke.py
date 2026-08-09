from pathlib import Path
import time
import numpy as np

import ray
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory


BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")
COMBINED = BASE / "outputs/r0269/018_rg070/out/combined_all_018_rg070"

EXPT = COMBINED / "combined_018_rg070.expt"
REFL = COMBINED / "combined_018_rg070.refl"


@ray.remote
class CombinedWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id

        t0 = time.perf_counter()
        self.experiments = ExperimentListFactory.from_json_file(
            str(EXPT),
            check_format=True,
        )
        self.reflections = flex.reflection_table.from_file(str(REFL))
        self.ids_np = self.reflections["id"].as_numpy_array()
        self.load_time = time.perf_counter() - t0

    def process_ids(self, experiment_ids):
        rows = []

        for experiment_id in experiment_ids:
            t0 = time.perf_counter()

            try:
                select_mask_np = self.ids_np == experiment_id
                select_mask = flex.bool(select_mask_np.tolist())
                refl_subset = self.reflections.select(select_mask)

                experiment = self.experiments[experiment_id]

                t_raw0 = time.perf_counter()
                raw = experiment.imageset.get_raw_data(0)
                raw_time = time.perf_counter() - t_raw0

                n_panels = len(raw)

                try:
                    experiment.imageset.clear_cache()
                except Exception:
                    pass

                rows.append(
                    {
                        "worker_id": self.worker_id,
                        "experiment_id": experiment_id,
                        "status": "OK",
                        "n_refl": len(refl_subset),
                        "n_panels": n_panels,
                        "load_time": self.load_time,
                        "raw_time": raw_time,
                        "total_time": time.perf_counter() - t0,
                        "error": "",
                    }
                )

            except Exception as e:
                rows.append(
                    {
                        "worker_id": self.worker_id,
                        "experiment_id": experiment_id,
                        "status": "ERROR",
                        "n_refl": -1,
                        "n_panels": -1,
                        "load_time": self.load_time,
                        "raw_time": -1,
                        "total_time": time.perf_counter() - t0,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        return rows


def chunk_list(xs, n_chunks):
    return [xs[i::n_chunks] for i in range(n_chunks)]


def main():
    num_workers = 4
    test_ids = list(range(20))

    ray.init(local_mode=True, include_dashboard=False)
    
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print(f"Testing experiment ids: {test_ids}")

    workers = [CombinedWorker.remote(i) for i in range(num_workers)]
    chunks = chunk_list(test_ids, num_workers)

    t0 = time.perf_counter()
    futures = [
        worker.process_ids.remote(chunk)
        for worker, chunk in zip(workers, chunks)
    ]

    results_nested = ray.get(futures)
    total_time = time.perf_counter() - t0

    results = [row for group in results_nested for row in group]

    print("")
    print("Results:")
    for row in sorted(results, key=lambda r: r["experiment_id"]):
        print(row)

    print("")
    print(f"Total Ray processing time: {total_time:.2f} sec")
    print(f"OK: {sum(r['status'] == 'OK' for r in results)}")
    print(f"ERROR: {sum(r['status'] == 'ERROR' for r in results)}")

    ray.shutdown()


if __name__ == "__main__":
    main()
