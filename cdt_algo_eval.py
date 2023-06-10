import multiprocessing
import uuid, math, time, datetime, pickle
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
from cdt.metrics import precision_recall, SID, SHD
from scm.dynamic_scm import DynamicSCM

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

def get_metric_mean_std_for_algo(scores, algo_name, metric):
    data = []
    for score_data in scores:
        data.append(score_data[algo_name][metric])
    mean = round(np.array(data).mean(), 2)
    std = round(np.array(data).std(), 2)
    return mean, std

def consolidate_scores(scores):
    for algo, metrics in scores.items():
        scores[algo]['aupr'] = (round(np.array(metrics["aupr"]).mean(), 2),
                                round(np.array(metrics["aupr"]).std(), 2))
        scores[algo]['sid'] = (round(np.array(metrics["sid"]).mean(), 2),
                                round(np.array(metrics["sid"]).std(), 2))
        scores[algo]['shd'] = (round(np.array(metrics["shd"]).mean(), 2),
                                round(np.array(metrics["aupr"]).std(), 2))
        scores[algo]['rt'] = (round(np.array(metrics["rt"]).mean(), 2),
                                round(np.array(metrics["rt"]).std(), 2))

def get_algo_metrics(orig_dag, algo_dag):
    aupr, _ = precision_recall(orig_dag, algo_dag)
    sid = SID(orig_dag, algo_dag)
    shd = SHD(orig_dag, algo_dag)
    return (aupr, sid, shd)

def get_algo_scores_data(algo_dags_and_runtimes, orig_dag):
    scores = {}
    for iter_data in algo_dags_and_runtimes.values():
        for algo_data in iter_data:
            algo_name = algo_data[0]
            algo_dag = algo_data[1]
            algo_metrics = get_algo_metrics(orig_dag, algo_dag)
            if algo_name in scores:
                scores[algo_name]["aupr"].append(algo_metrics[0])
                scores[algo_name]["sid"].append(algo_metrics[1]) 
                scores[algo_name]["shd"].append(algo_metrics[2]) 
                scores[algo_name]["rt"].append(algo_data[2])
            else:
                scores[algo_name]={}
                scores[algo_name]["aupr"] = [algo_metrics[0]]
                scores[algo_name]["sid"] = [algo_metrics[1]] 
                scores[algo_name]["shd"] = [algo_metrics[2]]
                scores[algo_name]["rt"] = [algo_data[2]]
                scores[algo_name]["dag"] = algo_dag
    consolidate_scores(scores)
    return scores
      
def train_algos_in_parallel(algo_meta_data, data, conf_name):
    uid = uuid.uuid4()
    context = f"{algo_meta_data['name']} for {conf_name} using {len(data)} samples - {uid}"
    log_progress(f"Start Training : {context}")
    
    start = time.time()
    algo = algo_meta_data["model"]()
    dag = algo.predict(data)
    run_time = round(time.time() - start, 2)
    
    log_progress(f"End Training : {context}")
    return ((algo_meta_data['name'],dag, run_time))
    
def get_timeout_value(conf, num_samples):
    if isinstance(conf["nodes"], list):
        num_nodes = sum(conf["nodes"])
    else:
        num_nodes = conf["nodes"]
    
    timeout = int(math.exp(int(math.log(num_nodes)) + 
                           int(math.log(math.sqrt((num_samples/10))))))
    return timeout

def get_data(num_samples, scm, conf_name, iter):
    while True:
        msg = f"{num_samples} samples for {conf_name}"
        log_progress
        try:
            log_msg = f"Iter {iter}: Starting data generation of {msg}"
            log_progress(log_msg)
            data = scm.sample(num_samples)
            log_msg = f"Iter {iter}: Completed data generation of {msg}"
            log_progress(log_msg)
            break
        except Exception as e:
            log_msg = f"Iter {iter}: EXCEPTION data generation. Trying again {msg}"
            log_progress(log_msg)
            continue
    return data

def train_algos_for_each_sample_size(scm, num_samples, conf):
    algo_dags_and_runtimes = {}
    num_processes = max(1, min(len(models), multiprocessing.cpu_count() - 1))
    timeout = get_timeout_value(conf, num_samples)

    for iter in range(num_iterations):
        data = get_data(num_samples, scm, conf["name"], iter)
        args_list = [(algo_meta_data, data, conf["name"]) for algo_meta_data in models]
        try:
            pool = multiprocessing.Pool(processes=num_processes)
            results = [pool.apply_async(train_algos_in_parallel, args=args) 
                        for args in args_list]
            results = [result.get(timeout=timeout) for result in results]
            algo_dags_and_runtimes[iter] = results
        except multiprocessing.TimeoutError:
            log_progress(f"**TIMEOUT after {timeout} secs in \
iteration {iter} for {conf['name']} using {num_samples}")

    return algo_dags_and_runtimes

def run_algo_training_for_various_sample_sizes(scm, conf):
    global row
    for num_samples in sample_sizes:
        algo_dags_and_runtimes = train_algos_for_each_sample_size(scm, 
                                                                  num_samples, conf)
        scores = get_algo_scores_data(algo_dags_and_runtimes, scm.dag)
        save_plots_to_file(plots_file, row, 1, scores, conf['name'], num_samples)  
        row = row + 1
        
def get_score_text(score_data):
    text = ""
    for metric, values in score_data.items():
        if metric != "dag":
            text += f"{metric}:{values}\n"
    return text

def save_plots_to_file(plots_file, row, col, scores, conf_name, num_samples=""):
    # Create a new Excel workbook to save and compate DAGs
    try:
        workbook = openpyxl.load_workbook(plots_file)
    except:
        workbook = openpyxl.Workbook()
    sheet = workbook.active

    for algo, score_data in scores.items():
        dag = score_data["dag"]
        pos = graphviz_layout(dag, prog="dot")
        fig, ax = plt.subplots(1, 1, figsize=(3,2), dpi=150)
        ax.set_title(f"{conf_name}_{algo}_{num_samples}")

        nx.draw(dag, pos=pos, ax=ax, with_labels=True, node_size=250, alpha=0.5)
        
        text_x = 0.65  # x position as a fraction of the figure width
        text_y = 0.05  # y position as a fraction of the figure heigh
        text = get_score_text(score_data)
        fig.text(text_x, text_y, text, fontsize=10, color='black')
        
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

def get_scm(config):
    input_nodes = config["nodes"]
    dSCM = eval(config["dSCM"])
    scm = dSCM.create(input_nodes)
    return scm

if __name__ == "__main__":
    for conf in configs:
        scm = get_scm(conf)
        save_plots_to_file(plots_file, row, 0, 
                           {"Original":{"dag": scm.dag}}, conf["name"])
        run_algo_training_for_various_sample_sizes(scm, conf)