import time
import logging
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
from dynamic_scm import DynamicSCM
from cdt.metrics import precision_recall, SID, SHD

# Init loging
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum logging level
    format='%(asctime)s - %(message)s',  # Define log message format
    filename='logs/cdt_algo_eval.log',  # Specify the log file name
    filemode='w'  # Set the file mode to 'write' (overwrite existing log)
)
logger = logging.getLogger(__name__)

# Create various DAG configurations
configs = [
    {
        "name" : "config_1",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[1], simple_operations={"+": 1})'
    },
    {
        "name" : "config_2",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[0.4,0.1,0.5])'
    },
    {
        "name" : "config_3",
        "nodes": 40,
        "dSCM" : 'DynamicSCM()'
    },
    {
        "name" : "config_4",
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
sample_sizes = [100, 1000, 10000, 20000, 50000, 100000]
total_steps = len(configs) * len(models) * len(sample_sizes)
step = 0

def log_progress(total_steps, step, config, sample, model):
    progress = round(step * 100/total_steps, 2)
    logger.info(f"Progress: {progress}% ({sample}, {config}, {model})")

def save_plots_to_file(plots_file, row, col, dag, config, algo, sample=""):
    # Create a new Excel workbook to save and compate DAGs
    try:
        workbook = openpyxl.load_workbook(plots_file)
    except:
        workbook = openpyxl.Workbook()
    sheet = workbook.active

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
    width, height = 30, 80
    sheet.column_dimensions[target_cell[0]].width = width
    sheet.row_dimensions[int(target_cell[1:])].height = height
    image.width = width * 6
    image.height = height * 1.3

    sheet.add_image(image, target_cell)
    workbook.save(plots_file)
    workbook.close()

def write_score_data_to_file(file_name, config, sample, algo, scores):
    line = f"{sample}, {config}, {algo}"
    for key in ["aupr", "sid", "shd", "duration"]:
        mean = round(np.array(scores[key]).mean(), 2)
        std = round(np.array(scores[key]).std(), 2)
        line = line + ", " + f"{mean}, {std}"
    errors = ";".join(scores["errors"])
    line = line + ", " + errors

    with open(file_name, "a") as f:
            f.writelines(line + "\n")
        
def get_algo_scores_and_dag(model, scm, size):
    dag = scm.dag
    algo = model()
    scores = {"aupr":[], "sid":[], "shd":[], "duration":[], "errors":[]}
    for _ in range(1, 6):
        data = scm.sample(size)
        
        try:
            start = time.time()
            prediction = algo.predict(data)
            end = time.time()

            aupr, _ = precision_recall(dag, prediction)
            sid_score = SID(dag, prediction)
            shd_score = SHD(dag, prediction)
            scores["aupr"].append(round(aupr, 3))
            scores["sid"].append(int(sid_score))
            scores["shd"].append(shd_score)
            scores["duration"].append(round(end - start, 2))
        except Exception as e:
            scores["aupr"].append(0)
            scores["sid"].append(0)
            scores["shd"].append(0)
            scores["duration"].append(0)
            scores["errors"].append(f"Error: {e}")
    return scores, prediction

def get_scm(config):
    input_nodes = config["nodes"]
    dSCM = eval(config["dSCM"])
    scm = dSCM.create(input_nodes)
    return scm

def init_scores_file(file_name):
    with open(file_name, "w+") as f:
        f.writelines(f"Sample, Config, Algo, AUPR, , SID, , SHD, , Duration, , Errors\n")
        f.writelines(f", , , mean, std, mean, std, mean, std, mean, std, \n")

init_scores_file(scores_file)
for c in configs:
    scm = get_scm(c)
    row, col = 0, 0
    row = row + 1
    save_plots_to_file(plots_file, row, col, scm.dag, c['name'], "Original")
    for s in sample_sizes:
        for m in models:
            scores, algo_dag = get_algo_scores_and_dag(m["model"], scm, s)
            write_score_data_to_file(scores_file, c['name'], s, m['name'], scores)
            
            col = col + 1
            save_plots_to_file(plots_file, row, col, algo_dag, c['name'], m['name'], s)
            
            step = step + 1
            log_progress(total_steps, step, c['name'], s, m['name'])