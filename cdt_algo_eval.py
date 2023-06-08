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

# Create various DAG configurations
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

# Create a disctionary of various CDT algorithms
models = [
    {
        "name" : "GS",
        "model" : GS
    },
    {
        "name" : "GES",
        "model" : GES
    },
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

scores_file = "logs/cdt_algo_scores.csv"
plots_file = "logs/cdt_algo_plots.xlsx"
log_file ='logs/cdt_algo_eval.log'
sample_sizes = [100, 1000, 10000, 20000, 50000, 100000]
total_steps = len(configs) * len(models) * len(sample_sizes)
step = 0

def log_progress(total_steps, step, config, sample, model, iter):
    progress = round(step * 100/total_steps, 2)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (f"Progress: {progress}% ({sample}, {config}, {model}, iter_{iter})")
    with open(log_file, "a+") as f:
        f.writelines(f"{ts} - {message}\n")

def save_plots_to_file(plots_file, row, col, scores, config, sample=""):
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
        ax.set_title(f"{config}_{algo}_{sample}")

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

def write_score_data_to_file(file_name, scores, config, sample):
    for algo, score_data in scores.items():
        line = f"{sample}, {config}, {algo}"
        for key in ["aupr", "sid", "shd", "duration"]:
            mean = round(np.array(score_data[key]).mean(), 2)
            std = round(np.array(score_data[key]).std(), 2)
            line = line + ", " + f"{mean}, {std}"

        errors = ";".join(score_data["errors"])
        line = line + ", " + errors
        with open(file_name, "a") as f:
                f.writelines(line + "\n")
        
def populate_algo_scores_and_dag(scores, algo_meta_data, dag, data):
    algo = algo_meta_data["model"]()
    algo_name = algo_meta_data["name"]
    try:
        start = time.time()
        prediction = algo.predict(data)
        end = time.time()

        aupr, _ = precision_recall(dag, prediction)
        sid_score = SID(dag, prediction)
        shd_score = SHD(dag, prediction)
        scores[algo_name]["aupr"].append(round(aupr, 3))
        scores[algo_name]["sid"].append(int(sid_score))
        scores[algo_name]["shd"].append(shd_score)
        scores[algo_name]["duration"].append(round(end - start, 2))
        scores[algo_name]["dag"] = prediction
    except Exception as e:
        scores[algo_name]["aupr"].append(0)
        scores[algo_name]["sid"].append(0)
        scores[algo_name]["shd"].append(0)
        scores[algo_name]["duration"].append(0)
        scores[algo_name]["errors"].append(f"Error: {e}")

def get_scm(config):
    input_nodes = config["nodes"]
    dSCM = eval(config["dSCM"])
    scm = dSCM.create(input_nodes)
    return scm

def init_scores_dict():
    scores = {}
    for m in models:
        scores[m["name"]] = {"aupr":[], "sid":[], "shd":[], 
                             "duration":[], "errors":[], "dag":None}
    return scores

def init_scores_file(file_name):
    with open(file_name, "w+") as f:
        f.writelines(f"Sample, Config, Algo, AUPR, , SID, , SHD, , Duration, , Errors\n")
        f.writelines(f", , , mean, std, mean, std, mean, std, mean, std, \n")

if __name__ == "__main__":
    init_scores_file(scores_file)
    for c in configs:
        scm = get_scm(c)
        orig_dag = scm.dag
        row = 0
        row = row + 1
        save_plots_to_file(plots_file, row, 0, {"Original":{"dag": orig_dag}}, c['name'])
        for s in sample_sizes:
            scores = init_scores_dict()
            for i in range(1, 6):
                data = scm.sample(s)
                for m in models:
                    populate_algo_scores_and_dag(scores, m, orig_dag, data)
                    step = step + 1
                    log_progress(total_steps, step, c['name'], s, m['name'], i)

            write_score_data_to_file(scores_file, scores, c['name'], s)
            save_plots_to_file(plots_file, row, 1, scores, c['name'], s)
                
            row = row + 1