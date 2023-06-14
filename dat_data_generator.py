import pickle, os, glob, yaml
import multiprocessing
from scmodels import SCM
import cdt_algo_eval_common_methods as cdt_cm

class DAGDataGenerator():
    def __init__(self, num_iterations, sample_sizes,
                 data_folder, conf_file) -> None:
        self.__num_iterations = num_iterations
        self.__sample_sizes = sample_sizes
        self.__data_folder = data_folder
        self.__conf_file = conf_file
        pass

    def generate_data(self, num_samples, iter, dists_file_path):
        folder = dists_file_path.split("/")[0]
        conf_name = dists_file_path.split("/")[1].split(".")[0]
        data_file = f"{folder}/{conf_name}_{num_samples}_{iter}.data"
        
        if os.path.exists(data_file):
            cdt_cm.log_progress(f"{data_file} already exists.")
            return data_file
        
        with open(dists_file_path, "rb") as file:
            scm_dists = pickle.load(file)
        scm = SCM(scm_dists)
        trials = 10
        while trials:
            msg = f"{num_samples} samples for {conf_name}"
            try:
                log_msg = f"Iter {iter}: Starting data generation of {msg}"
                cdt_cm.log_progress(log_msg)
                data = scm.sample(num_samples)

                with open(data_file, "wb") as file:
                    pickle.dump(data, file)

                log_msg = f"Iter {iter}: Completed data generation of {msg}"
                cdt_cm.log_progress(log_msg)
                break
            except Exception as e:
                log_msg = f"Iter {iter}: EXCEPTION in data generation. Trying again {msg}"
                cdt_cm.log_progress(log_msg)
                trials -= 1
                continue
        return data_file

    def get_args_for_dgp(self, dists_file_paths, 
                                             num_iterations, sample_sizes):
        args = [(num_samples, iter, path) for path in dists_file_paths 
                                                for iter in range(num_iterations) 
                                                    for num_samples in sample_sizes]
        return args

    def get_scm_and_scm_dists(self, config):
        input_nodes = config["nodes"]
        dSCM = eval(config["dSCM"])
        scm = dSCM.create(input_nodes)
        return scm, dSCM.get_scm_dists()

    def check_and_handle_dists_creation(self, dists_file, conf, data_folder):
        if not conf["force_data_generation"]:
            if os.path.exists(dists_file):
                cdt_cm.log_progress(f"{dists_file} already exists. \
To generate new distribution set 'force_data_generation' to true")
                return False

        pattern = f"{conf['name']}_*.data"
        data_files = glob.glob(os.path.join(data_folder, pattern))
        for file in data_files:
            os.remove(file)
        return True

    def generate_and_save_scm_dists(self, config_file, data_folder):
        dists_file_paths = []
        for conf in cdt_cm.get_dag_configs(self.__conf_file):
            dists_file = f"{data_folder}/{conf['name']}.dists"
            dists_file_paths.append(dists_file)

            create = self.check_and_handle_dists_creation(dists_file, 
                                                          conf, 
                                                          data_folder)
            if create:
                _, scm_dists = self.get_scm_and_scm_dists(conf)
                with open(dists_file, "wb") as file:
                    pickle.dump(scm_dists, file)   
        return dists_file_paths

    def generate_all_data(self):
        folder = self.__data_folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        dists_file_paths = self.generate_and_save_scm_dists(self.__conf_file, 
                                                            folder)
        args = self.get_args_for_dgp(dists_file_paths, 
                                     self.__num_iterations, 
                                     self.__sample_sizes)
        num_processes = max(1, min(len(args), multiprocessing.cpu_count() - 1))
        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.starmap(self.generate_data, args)
        results = [result for result in results]
        return dists_file_paths

