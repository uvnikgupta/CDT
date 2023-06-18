import multiprocessing
import time, random
from scmodels import SCM
from scm.dynamic_scm import DynamicSCM
import cdt_algo_eval_common_methods as cdt_cm
from algo_trainer import AlgoTrainer

"""
Methods:
    get_ASSIC : Get All Sample Siize and Iteration Combinations
    get_TAL : Get Training Arguments List
"""
class AlgoTrainerST2(AlgoTrainer):
    non_performant_algos = ["GIES", "PS"]

    def __init__(self, num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, status_file, data_folder):
        super().__init__(num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, status_file, data_folder)

    def get_ASSIC(self, algo_meta_data, conf, scm_dists):
        combinations = []
        for num_samples in self.sample_sizes:
            for iter in range(self.num_iterations):
                run_key = f"{algo_meta_data['name']}_{conf['name']}_{num_samples}_{iter}"
                if run_key not in self.completed_runs:
                    data_file = f"{self.data_folder}/{conf['name']}_{num_samples}_{iter}.data"
                    combinations.append((algo_meta_data, data_file, 
                                        conf["name"], scm_dists))
        return combinations

    def get_TAL(self, confs):
        args_list = []
        self.pupulate_run_status()
        for conf in confs:
            scm_dists = self.get_scm_dist(conf)
            conf_specific_algos = self.get_algos_for_training(conf)
            for algo_meta_data in conf_specific_algos:
                combs = self.get_ASSIC(algo_meta_data, conf, scm_dists)
                args_list.extend(combs)
        return args_list

    def start_training(self, st2_confs):
        timeout = 720
        fib_1, fib_2 = 34, 55
        num_processes = max(1, multiprocessing.cpu_count() - 2)
        
        while True:
            growth_factor = fib_1 + fib_2
            args_list = self.get_TAL(st2_confs)
            random.shuffle(args_list)
            if not args_list:
                break

            pool = multiprocessing.Pool(processes=num_processes)
            try:
                results = [pool.apply_async(self.train_algo, args=args) 
                                for args in args_list]
                results = [result.get(timeout=timeout) for result in results]
            except multiprocessing.TimeoutError:
                cdt_cm.log_progress(f"Timeout occured after {timeout} secs")
                timeout = timeout + growth_factor
                fib_1, fib_2 = fib_2, growth_factor

            pool.terminate()
            pool.join()
            time.sleep(15)