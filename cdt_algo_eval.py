import multiprocessing
from functools import partial
import uuid, math, time, datetime, pickle
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image
from io import BytesIO
from PIL import Image as PILImage
from cdt.causality.graph import (GS, GIES, PC, SAM, IAMB, Inter_IAMB, 
                                 Fast_IAMB, MMPC, LiNGAM, CAM, CCDr)
from cdt.metrics import precision_recall, SID, SHD
from scm.dynamic_scm import DynamicSCM
    
# Various DAG configurations
configs = [
    {
        "name" : "config_1",
        "nodes": [10,5,2],
        "dSCM" : 'DynamicSCM(min_parents=2, parent_levels_probs=[0.9, 0.3], \
            distribution_type=1, simple_operations={"+": 1})'
    },
    {
        "name" : "config_2",
        "nodes": [10,5,2],
        "dSCM" : 'DynamicSCM(min_parents=2, parent_levels_probs=[0.9, 0.3], \
            distribution_type=2, simple_operations={"+": 1})'
    },
    {
        "name" : "config_3",
        "nodes": [10,5,2],
        "dSCM" : 'DynamicSCM(min_parents=2, parent_levels_probs=[0.9, 0.3], \
            simple_operations={"+": 1})'
    },
    {
        "name" : "config_4",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[1], \
            distribution_type=1, simple_operations={"+": 1})'
    },
    {
        "name" : "config_5",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[1], \
            distribution_type=2, simple_operations={"+": 1})'
    },
    {
        "name" : "config_5",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, parent_levels_probs=[1], \
            simple_operations={"+": 1})'
    },
    {
        "name" : "config_6",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, \
            parent_levels_probs=[0.4,0.1,0.5], distribution_type=2)'
    },
    {
        "name" : "config_7",
        "nodes": [2,2,2,2,2,1],
        "dSCM" : 'DynamicSCM(min_parents=2, max_parents=2, \
            parent_levels_probs=[0.4,0.1,0.5])'
    },
    {
        "name" : "config_8",
        "nodes": 40,
        "dSCM" : 'DynamicSCM(distribution_type=1)'
    },
    {
        "name" : "config_9",
        "nodes": 40,
        "dSCM" : 'DynamicSCM()'
    },
    {
        "name" : "config_10",
        "nodes": [150, 50, 10, 3],
        "dSCM" : 'DynamicSCM(max_parents=20, complex_operations={})'
    }
]

# Various CDT algorithms
models = [
    {
        "name" : "GS", 
        "model" : GS,
        "parallel" : True
    },
    {
        "name" : "GIES",
        "model" : GIES,
        "parallel" : True
    },
    {
        "name" : "PC",
        "model" : PC,
        "parallel" : True
    },
    {
        "name" : "IAMB",
        "model" : IAMB,
        "parallel" : True
    },
    {
        "name" : "Fast_IAMB",
        "model" : Fast_IAMB,
        "parallel" : True
    },
    {
        "name" : "Inter_IAMB",
        "model" : Inter_IAMB,
        "parallel" : True
    },
    {
        "name" : "MMPC",
        "model" : MMPC,
        "parallel" : True
    },
    {
        "name" : "LiNGAM",
        "model" : LiNGAM,
        "parallel" : True
    },
    {
        "name" : "CAM",
        "model" : CAM,
        "parallel" : True
    },
    # {
    #     "name" : "CCDr",
    #     "model" : CCDr,
    #    "parallel" : True
    # },
    # {
    #     "name" : "SAM",
    #     "model" : SAM,
    #     "parallel" : False
    # },
]

# Global variables
plots_file = "logs/cdt_algo_plots.xlsx"
log_file ='logs/cdt_algo_eval.log'
sample_sizes = [100, 1000, 10000, 20000, 50000, 100000]
num_iterations = 5
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
    if not isinstance(algo_dag, Exception):
        aupr, _ = precision_recall(orig_dag, algo_dag)
        sid = SID(orig_dag, algo_dag)
        shd = SHD(orig_dag, algo_dag)
    else:
        aupr, _ = (0, 0)
        sid = np.inf
        shd = np.inf
    return (aupr, sid, shd)

def get_algo_scores_data(algo_dags_and_runtimes, orig_dag):
    log_progress("Calculation scores - Start")
    scores = {}
    for iter_data in algo_dags_and_runtimes.values():
        for algo_data in iter_data:
            algo_name = algo_data[0]
            algo_dag_file = algo_data[1]
            with open(algo_dag_file, "rb") as file:
                algo_dag = pickle.load(file)

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
    log_progress("Calculation scores - Done")
    return scores
      
def train_algo(algo_meta_data, data_file, conf_name, iter):
    uid = uuid.uuid4()
    with open(data_file, "rb") as file:
        data = pickle.load(file)

    context = f"{algo_meta_data['name']} for {conf_name} using {len(data)} samples - {uid}"
    log_progress(f"Start Training : {context}")
    
    start = time.time()
    algo = algo_meta_data["model"]()
    try:
        dag = algo.predict(data)
        run_time = round(time.time() - start, 2)
        log_progress(f"End Training : {context}")
    except Exception as e:
        dag = e
        run_time = 0
        log_progress(f"EXCEPTION in training {context}\n{e}")

    dag_file_name = f"{algo_meta_data['name']}_{iter}.dag"
    with open(dag_file_name, "wb") as file:
        pickle.dump(dag, file)

    return ((algo_meta_data['name'], dag_file_name, run_time))
    
def get_timeout_value(conf, num_samples):
    if isinstance(conf["nodes"], list):
        num_nodes = sum(conf["nodes"])
    else:
        num_nodes = conf["nodes"]
    
    timeout = int(math.exp(int(math.log(num_nodes)) + 
                           int(math.log(math.sqrt((num_samples/10))))))
    return timeout

def train_algos_for_sample_size(data_file_names, num_samples, conf):
    algo_dags_and_runtimes = {}
    num_processes = max(1, min(len(models), multiprocessing.cpu_count() - 1))
    timeout = get_timeout_value(conf, num_samples)

    for iter in range(num_iterations):
        log_progress(f"\n**** Starting training iteration {iter} for \
{num_samples} samples ****")
        data_file = data_file_names[iter]
        args_list = [(algo_meta_data, data_file, conf["name"], iter) 
                     for algo_meta_data in models if algo_meta_data["parallel"] == True]
        try:
            pool = multiprocessing.Pool(processes=num_processes)
            results = [pool.apply_async(train_algo, args=args) 
                        for args in args_list]
            results = [result.get(timeout=timeout) for result in results]

            # Train the algos that cannot be run in parallel. These are the 
            # NN based algos that spawn their own process hence cannot be run
            # from within an already spawned process
            for algo_meta_data in models:
                if not algo_meta_data["parallel"]:
                    result = train_algo(algo_meta_data, data_file, conf["name"])
                    results.append(result)

            algo_dags_and_runtimes[iter] = results
        except multiprocessing.TimeoutError:
            log_progress(f"**TIMEOUT after {timeout} secs in \
iteration {iter} for {conf['name']} using {num_samples}")
            
        log_progress(f"\n==== Completed training iteration {iter} for \
{num_samples} samples ====")
    return algo_dags_and_runtimes

def generate_data(num_samples, scm_dists, conf_name, iter):
    from scmodels import SCM
    
    scm = SCM(scm_dists)
    trials = 10
    while trials:
        msg = f"{num_samples} samples for {conf_name}"
        try:
            log_msg = f"Iter {iter}: Starting data generation of {msg}"
            log_progress(log_msg)
            data = scm.sample(num_samples)

            file_name = f"data_{iter}.pkl"
            with open(file_name, "wb") as file:
                pickle.dump(data, file)

            log_msg = f"Iter {iter}: Completed data generation of {msg}"
            log_progress(log_msg)
            break
        except Exception as e:
            log_msg = f"Iter {iter}: EXCEPTION in data generation. Trying again {msg}"
            log_progress(log_msg)
            trials -= 1
            continue
    return file_name

def populate_cdt_algos_scores_for_config(scm_dists, conf):
    global row
    iters = [(i,) for i in range(num_iterations)]
    num_processes = max(1, min(num_iterations, multiprocessing.cpu_count() - 1))

    # Run algo training for various sample sizes for the config
    for num_samples in sample_sizes:
        # Run parallel data generation for all iterations of algo training
        # Cannot generate the data while training the algos in parallel as
        # each algorithm will then have a different set of data!
        partial_worker = partial(generate_data, num_samples, scm_dists, conf["name"])
        pool = multiprocessing.Pool(processes=num_processes)
        file_names = pool.starmap(partial_worker, iters)
        pool.close()
        pool.join()

        # Train CDT algos and get their DAGs and run durations
        algo_dags_and_runtimes = train_algos_for_sample_size(file_names, 
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
    log_progress("Saving DAGs - Start")
    try:
        workbook = openpyxl.load_workbook(plots_file)
    except:
        workbook = openpyxl.Workbook()
    sheet = workbook.active

    for algo, score_data in scores.items():
        dag = score_data["dag"]
        fig, ax = plt.subplots(1, 1, figsize=(3,2), dpi=150)
        ax.set_title(f"{conf_name}_{algo}_{num_samples}")

        if not isinstance(dag, Exception):
            pos = graphviz_layout(dag, prog="dot")
            nx.draw(dag, pos=pos, ax=ax, with_labels=True, node_size=250, alpha=0.5)
            text = get_score_text(score_data)
            fig.text(0.65, 0.05, text, fontsize=10, color='black')
        else:
            text = dag
            fig.text(0.2, 0.1, text, fontsize=8, color='red')
        
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
    log_progress("Saving DAGs - Done")

def get_scm(config):
    input_nodes = config["nodes"]
    dSCM = eval(config["dSCM"])
    scm = dSCM.create(input_nodes)
    return scm, dSCM.get_scm_dists()

if __name__ == "__main__":
    for conf in configs:
        scm, scm_dists = get_scm(conf)
        save_plots_to_file(plots_file, row, 0, 
                           {"Original":{"dag": scm.dag}}, conf["name"])
        populate_cdt_algos_scores_for_config(scm_dists, conf)