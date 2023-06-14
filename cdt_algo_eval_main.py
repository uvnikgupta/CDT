import os, sys
from dat_data_generator import DAGDataGenerator
from algo_trainer_st1 import AlgoTrainerST1

conf_file = "dag_generation_configs.yml"
data_folder = "data_cdt_algo_eval"
cdt_algos_file = "cdt_algos.yml"
plots_file = "logs/plots_cdt_algo.xlsx"

sample_sizes = [1000, 10000, 20000, 50000, 100000]
num_iterations = 3

if __name__ == "__main__":
    arguments = sys.argv

    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    if len(arguments) == 1:
        for filename in os.listdir("logs"):
            file_path = os.path.join("logs", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        data_generator = DAGDataGenerator(num_iterations, sample_sizes,
                                          data_folder, conf_file)
        data_generator.generate_all_data()

    train_st1 = AlgoTrainerST1(num_iterations, sample_sizes, 
                               plots_file, conf_file, cdt_algos_file, 
                               data_folder)
    train_st1.start_training()