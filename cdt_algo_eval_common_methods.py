import datetime
import yaml

log_file ='logs/cdt_algo_eval.log'

def log_progress(message):
    with open(log_file, "a+") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.writelines(f"{ts} - {message}\n")

def get_dag_configs(conf_file):
    with open(conf_file,"r") as file:
        configs = yaml.safe_load(file)
    return configs
        
    