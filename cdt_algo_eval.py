import multiprocessing
from functools import partial
import uuid
import math
import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image
from io import BytesIO
from PIL import Image as PILImage
from cdt.causality.graph import (GS, GES, GIES, PC, SAM, SAMv1, IAMB, Inter_IAMB, 
                                 Fast_IAMB, MMPC, LiNGAM, CAM, CCDr)
from scm.dynamic_scm import DynamicSCM
from cdt.metrics import precision_recall, SID, SHD

# Exception class for catching exception while generating samples
class GenerateSampleException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Various DAG configurations
configs = [
    {
        "name" : "config_1",
        "nodes": [10,5,2],
        "dSCM" : 'DynamicSCM(min_parents=2, parent_levels_probs=[0.9, 0.3], simple_operations={"+": 1})'
    },
    {
        "name" : "config_2",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[1], simple_operations={"+": 1})'
    },
    {
        "name" : "config_3",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[0.4,0.1,0.5])'
    },
    {
        "name" : "config_4",
        "nodes": 40,
        "dSCM" : 'DynamicSCM()'
    },
    {
        "name" : "config_5",
        "nodes": [150, 50, 10, 3],
        "dSCM" : 'DynamicSCM(max_parents=20, complex_operations={})'
    }
]

# Various CDT algorithms
models = [
    {
        "name" : "GS", 
        "model" : GS
    },
    # GES algorithm seems to have a bug where it goes into infinite loop sometimes
    # Moreover it seems that GES and GIES are essentially doing the same thing! 
    # {
    #     "name" : "GES",
    #     "model" : GES
    # },
    {
        "name" : "GIES",
        "model" : GIES
    },
    {
        "name" : "PC",
        "model" : PC
    },
    {
        "name" : "IAMB",
        "model" : IAMB
    },
    {
        "name" : "Fast_IAMB",
        "model" : Fast_IAMB
    },
    {
        "name" : "Inter_IAMB",
        "model" : Inter_IAMB
    },
    {
        "name" : "MMPC",
        "model" : MMPC
    },
    # {
    #     "name" : "CAM",
    #     "model" : CAM
    # },
    # {
    #     "name" : "LiNGAM",
    #     "model" : LiNGAM
    # },
    # {
    #     "name" : "CCDr",
    #     "model" : CCDr
    # },
    # {
    #     "name" : "SAM",
    #     "model" : SAM
    # },
    # {
    #     "name" : "SAMv1",
    #     "model" : SAMv1
    # },
    
]

# Global variables
scores_file = "logs/cdt_algo_scores.csv"
plots_file = "logs/cdt_algo_plots.xlsx"
log_file ='logs/cdt_algo_eval.log'
sample_sizes = [100, 1000, 10000, 20000, 50000, 100000]
num_iterations = 5
total_steps = len(configs) * len(models) * len(sample_sizes) * num_iterations
step = 0
row = 1
   
def log_progress(message):
    with open(log_file, "a+") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.writelines(f"{ts} - {message}\n")

def save_plots_to_file(plots_file, row, col, scores, conf_name, sample=""):
    # Create a new Excel workbook to save and compate DAGs
    try:
        workbook = openpyxl.load_workbook(plots_file)
    except:
        workbook = openpyxl.Workbook()
    sheet = workbook.active

    for algo, score_data in scores[-1].items():
        dag = score_data["dag"]
        pos = graphviz_layout(dag, prog="dot")
        fig, ax = plt.subplots(1, 1, figsize=(3,2), dpi=150)
        ax.set_title(f"{conf_name}_{algo}_{sample}")

        nx.draw(dag, pos=pos, ax=ax, with_labels=True, node_size=250, alpha=0.5)
        fig.savefig("plot.png")
        plt.close(fig)
        
        pil_image = PILImage.open('plot.png')
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        image = Image(image_bytes)

        target_cell = chr(65 + col) + str(row)
        width, height = 35, 110
        sheet.column_dimensions[target_cell[0]].width = width
        sheet.row_dimensions[int(target_cell[1:])].height = height
        image.width = width * 6.8
        image.height = height * 1.3

        sheet.add_image(image, target_cell)
        col = col + 1
    workbook.save(plots_file)
    workbook.close()

def get_metric_mean_std_for_algo(scores, algo_name, metric):
    data = []
    for score_data in scores:
        data.append(score_data[algo_name][metric])
    mean = round(np.array(data).mean(), 2)
    std = round(np.array(data).std(), 2)
    return mean, std

def write_scores_to_file(file_name, scores, conf_name, num_samples):
    for algo_meta_data in models:
        algo_name = algo_meta_data["name"]
        line = f"{conf_name}, {num_samples}, {algo_name}"
        for metric in ["aupr", "sid", "shd", "duration"]:
            mean, std = get_metric_mean_std_for_algo(scores, algo_name, metric)
            line = line + ", " + f"{mean}, {std}"

        # errors = ";".join(score_data["errors"])
        # line = line + ", " + errors
        with open(file_name, "a") as f:
                f.writelines(line + "\n")
        
def populate_algo_scores_and_dag(scores, algo_meta_data, orig_dag, data):
    algo = algo_meta_data["model"]()
    algo_name = algo_meta_data["name"]
    try:
        start = time.time()
        prediction = algo.predict(data)
        end = time.time()

        aupr, _ = precision_recall(orig_dag, prediction)
        sid_score = SID(orig_dag, prediction)
        shd_score = SHD(orig_dag, prediction)
        scores[algo_name]["aupr"] = round(aupr, 3)
        scores[algo_name]["sid"] = int(sid_score)
        scores[algo_name]["shd"] = shd_score
        scores[algo_name]["duration"] = round(end - start, 2)
        scores[algo_name]["dag"] = prediction
    except Exception as e:
        scores[algo_name]["aupr"] = 0
        scores[algo_name]["sid"] = 0
        scores[algo_name]["shd"] = 0
        scores[algo_name]["duration"] = 0
        scores[algo_name]["errors"] = f"Error: {e}"

def execute_algos(scm, scores, data, conf_name):
    for algo_meta_data in models:
        pid = uuid.uuid4()
        context = f"{algo_meta_data['name']}_{len(data)}_{conf_name} : {pid}"
        log_progress(f"Strating {context}")
        populate_algo_scores_and_dag(scores, algo_meta_data, scm.dag, data)
        log_progress(f"Completed {context}")

def init_scores_dict():
    scores = {}
    for m in models:
        scores[m["name"]] = {"aupr":None, "sid":None, "shd":None, 
                             "duration":None, "errors":None, "dag":None}
    return scores

def get_scm(config):
    input_nodes = config["nodes"]
    dSCM = eval(config["dSCM"])
    scm = dSCM.create(input_nodes)
    return scm

def get_algo_scores_for_each_sample_size(conf, num_samples):
    scm = get_scm(conf)
    scores = init_scores_dict()
    try:
        data = scm.sample(num_samples)
    except Exception as e:
        raise GenerateSampleException(e)
    
    execute_algos(scm, scores, data, conf["name"])
    return scores

def get_timeout_value(conf, num_samples):
    timeout = 1000
    if isinstance(conf["nodes"], list):
        num_nodes = sum(conf["nodes"])
    else:
        num_nodes = conf["nodes"]
    
    timeout = int(math.exp(int(math.log(num_nodes)) + int(math.log(math.sqrt(num_samples)))))
    return timeout

def execute_config_for_various_sample_sizes_in_parallel(conf):
    global row
    num_processes = max(1, min(num_iterations ,multiprocessing.cpu_count() - 1))
    for num_samples in sample_sizes:
        timeout = get_timeout_value(conf, num_samples)
        pool = multiprocessing.Pool(processes=num_processes)
        arguments = [(conf, num_samples) for _ in range(1, num_iterations + 1)]
        try:
            results = [pool.apply_async(get_algo_scores_for_each_sample_size, args=args) 
                       for args in arguments]
            scores = [result.get(timeout=timeout) for result in results]
            write_scores_to_file(scores_file, scores, conf['name'], num_samples)
            save_plots_to_file(plots_file, row, 1, scores, conf['name'], num_samples)  
            row = row + 1
        except GenerateSampleException as e:
            msg = f"EXCEPTION in {conf['name']}_{num_samples}\n{e}"
            log_progress(msg)
        except multiprocessing.TimeoutError as e:
            num_nodes = sum(conf["nodes"]) if isinstance(conf["nodes"], list) else conf["nodes"]
            msg = f"TIMEOUT after {timeout} secs in {conf['name']}_{num_samples} \
for {num_nodes} nodes in the configuration\n{e}"
            log_progress(msg)

def init_scores_file(file_name):
    with open(file_name, "w+") as f:
        f.writelines(f"Config, Sample, Algo, AUPR, , SID, , SHD, , Duration, , Errors\n")
        f.writelines(f", , , mean, std, mean, std, mean, std, mean, std, \n")

if __name__ == "__main__":
    init_scores_file(scores_file)
    for conf in configs:
        scm = get_scm(conf)
        save_plots_to_file(plots_file, row, 0, [{"Original":{"dag": scm.dag}}], conf['name'])
        execute_config_for_various_sample_sizes_in_parallel(conf)