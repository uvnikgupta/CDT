import os, sys
import numpy as np
from dag_data_generator import DAGDataGenerator
from algo_trainer_st1 import AlgoTrainerST1
from algo_trainer_st2 import AlgoTrainerST2
import cdt_algo_eval_common_methods as cdt_cm

conf_file = "dag_generation_configs.yml"
data_folder = "data_cdt_algo_eval"
cdt_algos_file = "cdt_algos.yml"
plots_file = "logs/plots_cdt_algo.xlsx"
status_file = "logs/training_status.yml"

sample_sizes = [1000, 10000, 20000, 50000, 100000]
num_iterations = 3

def train_algos(conf_file, num_iterations, sample_sizes, 
                plots_file, cdt_algos_file, data_folder):
    
    st1_confs, st2_confs = [], []
    for conf in cdt_cm.get_dag_configs(conf_file):
        num_nodes = sum(np.ravel(np.array([conf["nodes"]])))
        if num_nodes <= 30:
            st1_confs.append(conf)
        else:
            st2_confs.append(conf)
    
    trainer = AlgoTrainerST1(num_iterations, sample_sizes,
                             plots_file, cdt_algos_file, 
                             status_file, data_folder)        
    trainer.start_training(st1_confs)

    trainer = AlgoTrainerST2(num_iterations, sample_sizes,
                                plots_file, cdt_algos_file, 
                                status_file, data_folder)
    trainer.start_training(st2_confs)

def clean_folders(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    arguments = sys.argv

    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    if len(arguments) == 1:
        clean_folders("logs")
        data_generator = DAGDataGenerator(num_iterations, sample_sizes,
                                          data_folder, conf_file)
        data_generator.generate_all_data()

    train_algos(conf_file, num_iterations, sample_sizes, 
                plots_file, cdt_algos_file, data_folder)