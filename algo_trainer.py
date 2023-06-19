import os, yaml, time, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cdt.causality.graph import (GS, GIES, PC, SAM, IAMB, Inter_IAMB, 
                                 Fast_IAMB, MMPC, LiNGAM, CAM, CCDr)
from cdt.metrics import precision_recall, SID, SHD
from scmodels import SCM
import cdt_algo_eval_common_methods as cdt_cm

class AlgoTrainer():
    plots_sheet_name = "plots"
    config_sheet_name = "configs"
    metrics = {
            "aupr":'round(float(precision_recall(orig_dag, algo_dag)[0]), 2)', 
            "sid":'int(SID(orig_dag, algo_dag))',
            "shd":'int(SHD(orig_dag, algo_dag))',
            "rt":'round(time.time() - start, 2)'
            }
    
    def __init__(self, num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, status_file, data_folder):
        self.num_iterations = num_iterations
        self.sample_sizes = sample_sizes
        self.plots_file = plots_file
        self.cdt_algos_file = cdt_algos_file
        self.status_file = status_file
        self.data_folder = data_folder

    def pupulate_run_status(self):
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as file:
                status = yaml.safe_load(file)
                self.completed_runs = list(status.keys())
        else:
            self.completed_runs = []

    def get_algos_for_training(self, conf):
        with open(self.cdt_algos_file) as file:
            models = yaml.safe_load(file)

        algos = []
        for algo_meta_data in models:
            if "type" in algo_meta_data:
                if "type" in conf and algo_meta_data["type"] == conf["type"]:
                    # add type specifc algos
                    algos.append(algo_meta_data)
                elif "type" not in conf and algo_meta_data["type"] == 1:
                    # if conf is of mixed type add continuous alogs
                    algos.append(algo_meta_data)
            else: # add common algos
                algos.append(algo_meta_data)
        return algos
    
    def get_scm_dist(self, conf):
        dists_file = f"{self.data_folder}/{conf['name']}.dists"
        with open(dists_file, "rb") as file:
            scm_dists = pickle.load(file)
        return scm_dists

    def update_status(self, dag_file_name, metrics):
        key = dag_file_name.split("/")[1].split(".")[0]
        value = metrics
        value["dag"] = dag_file_name
        data = { key:value }
        with open(self.status_file, "a") as file:
            yaml.dump(data, file)
        
    def get_dag_file_name(self, algo_meta_data, data_file):
        if "/" in data_file:
            prefix = data_file.split("/")[1].split(".")[0]
        else:
            prefix = data_file.split(".")[0]
        dag_file_name = f"temp/{algo_meta_data['name']}_{prefix}.dag"
        return dag_file_name
    
    def get_algo_metrics(self, algo_name, orig_dag, algo_dag, start):
        retval ={}
        if algo_dag is not None:
            for metric, formula in self.metrics.items():
                if "sam" in algo_name.lower() and "sid" in metric.lower():
                    # Calculating SID for DAGs generated by SAM does not finish
                    # in reasonable time. Hence setting this metric to inf for SAM
                    retval[metric] = np.inf
                else:
                    retval[metric] = eval(formula)
        else:
            for metric, formula in self.metrics.items():
                retval[metric] = 0
        return retval

    def scale_data(self, data):
        columns = data.columns
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data=data, columns=columns)
        return data

    def train_algo(self, algo_meta_data, data_file, conf_name, orig_scm_dists):
        scm = SCM(orig_scm_dists)
        algo_name = algo_meta_data["name"]
        
        with open(data_file, "rb") as file:
            data = pickle.load(file)

        if "scale_data" in algo_meta_data and algo_meta_data["scale_data"]:
            data = self.scale_data(data)

        iter = data_file.split("_")[-1].split(".")[0]
        context = f"{algo_name}_{conf_name}_{len(data)}_{iter}"
        cdt_cm.log_progress(f"Start Training : {context}")
        
        start = time.time()
        algo = eval(algo_meta_data["model"])
        try:
            algo_dag = algo.predict(data)
            metrics = self.get_algo_metrics(algo_name, scm.dag, algo_dag, start)
            cdt_cm.log_progress(f"End Training : {context}")
        except Exception as e:
            algo_dag = f"EXCEPTION in training {context}"
            metrics = self.get_algo_metrics(algo_name, None, None, 0)
            cdt_cm.log_progress(f"EXCEPTION in training {context}\n{e}")

        dag_file_name = self.get_dag_file_name(algo_meta_data, data_file)
        with open(dag_file_name, "wb") as file:
            pickle.dump(algo_dag, file)

        self.update_status(dag_file_name, metrics)
        return ((algo_meta_data['name'], dag_file_name, metrics))
        
    

    