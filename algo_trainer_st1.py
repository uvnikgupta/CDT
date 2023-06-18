import multiprocessing
import pickle, math
import numpy as np
from scmodels import SCM
from scm.dynamic_scm import DynamicSCM
import cdt_algo_eval_common_methods as cdt_cm
from algo_trainer import AlgoTrainer

class AlgoTrainerST1(AlgoTrainer):
    def __init__(self, num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, status_file, data_folder):
        super().__init__(num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, status_file, data_folder)

    def update_scores(self, scores, results):
        for result in results:
            algo_name = result[0]
            with open(result[1], "rb") as file:
                dag = pickle.load(file)
            
            if algo_name in scores.keys():
                scores[algo_name]['dag'] = dag
                for metric in self.metrics:
                    scores[algo_name][metric].append(result[2][metric])
            else:
                scores[algo_name] = {}
                scores[algo_name]["dag"] = dag
                for metric in self.metrics:
                    scores[algo_name][metric]= [result[2][metric]]

    def train_non_parallable_algos(self, algos, data_file, 
                                   conf, orig_scm_dists, results):
        # Then train the algos that cannot be run in parallel. These are 
        # the NN based algos that spawn their own process hence cannot be 
        # run from within an already spawned process
        results = []
        for algo_meta_data in algos:
            if not algo_meta_data["parallel"]:
                result = self.train_algo(algo_meta_data, data_file, 
                                         conf["name"], orig_scm_dists)
                results.append(result)
        return results

    def prune_completed_runs(self):
        pruned_runs = set()
        for run in self.completed_runs:
            pruned_runs.add("_".join(run.split("_")[:-1]) )
        return pruned_runs
                   
    def get_training_args_list(self, algos, data_file, conf, orig_scm_dists):
        # Create the argument list for training algos in parallel
        args_list = []
        num_samples = data_file.split("_")[-2]
        pruned_runs = self.prune_completed_runs()
        for algo_meta_data in algos:
            run_key = f"{algo_meta_data['name']}_{conf['name']}_{num_samples}"
            if (algo_meta_data["parallel"] == True) and (run_key not in pruned_runs):
                args_list.append((algo_meta_data, data_file, conf["name"], orig_scm_dists))
        return args_list

    def get_timeout_value(self, conf, num_samples):
        if isinstance(conf["nodes"], list):
            num_nodes = sum(conf["nodes"])
        else:
            num_nodes = conf["nodes"]
        
        timeout = int(math.exp(int(math.log(num_nodes)) + 
                            int(math.log(math.sqrt((num_samples))))))
        return timeout

    def train_algos_for_sample_size(self,algos, num_samples, 
                                    conf, orig_scm_dists):
        scores ={}
        num_processes = max(1, min(len(algos), multiprocessing.cpu_count() - 1))
        timeout = self.get_timeout_value(conf, num_samples)

        for iter in range(self.num_iterations):
            cdt_cm.log_progress(f"\n**** Starting training iteration {iter} for \
{num_samples} samples ****")
            data_file = f"{self.data_folder}/{conf['name']}_{num_samples}_{iter}.data"
            args_list = self.get_training_args_list(algos, data_file, 
                                                    conf, orig_scm_dists)

            pool = multiprocessing.Pool(processes=num_processes)
            try:
                # First Train algos in parallel with timeout
                results = [pool.apply_async(self.train_algo, args=args) 
                            for args in args_list]
                results = [result.get(timeout=timeout) for result in results]
                result = self.train_non_parallable_algos(algos, data_file, 
                                                         conf, orig_scm_dists, 
                                                         results)
                if result: results.append(result)
                self.update_scores(scores, results)
            except multiprocessing.TimeoutError:
                cdt_cm.log_progress(f"**TIMEOUT after {timeout} secs in \
iteration {iter} for {conf['name']} using {num_samples}")

            pool.terminate()    
            pool.join()
            cdt_cm.log_progress(f"\n==== Completed training iteration {iter} for \
    {num_samples} samples ====")
        return scores

    def train_algos_for_config_and_save_results(self, scm_dists, conf):
        algos = self.get_algos_for_training(conf)

        # Run algo training for various sample sizes for the config
        for num_samples in self.sample_sizes:
            # Train CDT algos and get their DAGs and run durations
            self.train_algos_for_sample_size(algos, num_samples, 
                                                      conf, scm_dists)
            
    def start_training(self, confs):
        for conf in confs:
            scm_dists = self.get_scm_dist(conf)
            self.pupulate_run_status()
            self.train_algos_for_config_and_save_results(scm_dists, conf)

