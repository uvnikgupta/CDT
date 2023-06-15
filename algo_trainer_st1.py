import multiprocessing
import uuid, time, pickle, os
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image
from io import BytesIO
from PIL import Image as PILImage
from cdt.causality.graph import (GS, GIES, PC, SAM, IAMB, Inter_IAMB, 
                                 Fast_IAMB, MMPC, LiNGAM, CAM, CCDr)
from cdt.metrics import precision_recall, SID, SHD
from scmodels import SCM
from scm.dynamic_scm import DynamicSCM
import cdt_algo_eval_common_methods as cdt_cm
from algo_trainer import AlgoTrainer

plt.switch_backend('agg')

class AlgoTrainerST1(AlgoTrainer):
    def __init__(self, num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, data_folder):
        super().__init__(num_iterations, sample_sizes, 
                 plots_file, cdt_algos_file, data_folder)

    def consolidate_scores(self, scores):
        for algo in scores.keys():
            for metric in self.metrics.keys():
                mean = round(np.array(scores[algo][metric]).mean(), 2)
                std = round(np.array(scores[algo][metric]).std(), 2)
                scores[algo][metric] = (mean, std)

    def update_scores(self, scores, results):
        for result in results:
            algo_name = result[0]
            with open(result[1], "rb") as file:
                dag = pickle.load(file)
            scores[algo_name]['dag'] = dag
            for metric in self.metrics:
                scores[algo_name][metric].append(result[2][metric])

    def train_non_parallable_algos(self, algos, data_file, 
                                   conf, orig_scm_dists, results):
        # Then train the algos that cannot be run in parallel. These are 
        # the NN based algos that spawn their own process hence cannot be 
        # run from within an already spawned process
        results = []
        for algo_meta_data in algos:
            if not algo_meta_data["parallel"]:
                result = self.train_algo(algo_meta_data, data_file, 
                                         conf["name"], orig_scm_dists)
                results.append(result)
        return results

    def train_algo(self, algo_meta_data, data_file, conf_name, orig_scm_dists):
        uid = uuid.uuid4()
        scm = SCM(orig_scm_dists)
        algo_name = algo_meta_data["name"]
        
        with open(data_file, "rb") as file:
            data = pickle.load(file)

        if "scale_data" in algo_meta_data and algo_meta_data["scale_data"]:
            data = self.scale_data(data)

        context = f"{algo_name} for {conf_name} using {len(data)} samples - {uid}"
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

        dag_file_name = f"temp/{algo_meta_data['name']}.dag"
        with open(dag_file_name, "wb") as file:
            pickle.dump(algo_dag, file)

        return ((algo_meta_data['name'], dag_file_name, metrics))
        
    def get_training_args_list(self, algos, data_file, conf, orig_scm_dists):
        # Create the argument list for training algos in parallel
        args_list = []
        for algo_meta_data in algos:
            if algo_meta_data["parallel"] == True:
                args_list.append((algo_meta_data, data_file, conf["name"], orig_scm_dists))
        return args_list

    def init_scores(self, models):
        scores = {}
        for algo_meta_data in models:
            algo_name = algo_meta_data["name"]
            scores[algo_name] = {}
            scores[algo_name]["dag"] = None
            for metric in self.metrics:
                scores[algo_name][metric] = []
        return scores

    def train_algos_for_sample_size(self,algos, num_samples, 
                                    conf, orig_scm_dists):
        scores = self.init_scores(algos)
        num_processes = max(1, min(len(algos), multiprocessing.cpu_count() - 1))
        timeout = self.get_timeout_value(conf, num_samples)

        for iter in range(self.num_iterations):
            cdt_cm.log_progress(f"\n**** Starting training iteration {iter} for \
    {num_samples} samples ****")
            data_file = f"{self.data_folder}/{conf['name']}_{num_samples}_{iter}.data"
            args_list = self.get_training_args_list(algos, data_file, 
                                                    conf, orig_scm_dists)
            try:
                # First Train algos in parallel with timeout
                pool = multiprocessing.Pool(processes=num_processes)
                results = [pool.apply_async(self.train_algo, args=args) 
                            for args in args_list]
                results = [result.get(timeout=timeout) for result in results]
                result = self.train_non_parallable_algos(algos, data_file, 
                                                         conf, orig_scm_dists, 
                                                         results)
                if result: results.append(result)
                self.update_scores(scores, results)
            except multiprocessing.TimeoutError:
                cdt_cm.log_progress(f"**TIMEOUT after {timeout} secs in \
iteration {iter} for {conf['name']} using {num_samples}")
                
            cdt_cm.log_progress(f"\n==== Completed training iteration {iter} for \
    {num_samples} samples ====")
        return scores

    def train_algos_for_config_and_save_results(self, scm_dists, conf):
        algos = self.get_algos_for_training(conf)

        # Run algo training for various sample sizes for the config
        for num_samples in self.sample_sizes:
            # Train CDT algos and get their DAGs and run durations
            scores = self.train_algos_for_sample_size(algos, num_samples, 
                                                      conf, scm_dists)
            self.consolidate_scores(scores)
            self.save_plots_and_scores(1, conf, scores, num_samples)  
            self.row = self.row + 1
            
    def save_config_to_xl(self, configs_sheet, conf):
        row = 1
        while configs_sheet.cell(row=row, column=1).value is not None:
            row +=1
        configs_sheet.cell(row=row, column=1, value=str(conf))    

    def save_plots_and_scores_to_xl(self, plots_sheet, col, scores, 
                                    conf, num_samples=""):
        for algo, score_data in scores.items():
            # Since all the DAGs have the same number of nodes with the 
            # same names, node colors will of any one DAG will be applicable
            # for all DAGs. Since the only way to access an item in a python 
            # dict is via the key, hence this sing loop for loop
            node_colors = self.get_node_colors(score_data["dag"])
            break

        for algo, score_data in scores.items():
            title = f"{conf['name']}_{algo}_{num_samples}"
            dag = score_data["dag"]
            fig, ax = plt.subplots(1, 1, figsize=(3,2), dpi=150)
            ax.set_title(title)

            if isinstance(dag, nx.classes.digraph.DiGraph):
                pos = graphviz_layout(dag, prog="dot")
                nx.draw(dag, pos=pos, ax=ax, with_labels=True, node_size=150, 
                        node_color=node_colors, font_size=7, alpha=0.5)
                text = self.get_score_text(score_data)
                fig.text(0.65, 0.05, text, fontsize=10, color='black')
            else:
                text = dag
                fig.text(0.2, 0.2, text, fontsize=8, color='red')
            
            fig.savefig("plot.png")
            plt.close(fig)
            
            pil_image = PILImage.open('plot.png')
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            image = Image(image_bytes)

            target_cell = chr(65 + col) + str(self.row)
            plots_sheet[target_cell] = title
            width, height = 35, 110
            plots_sheet.column_dimensions[target_cell[0]].width = width
            plots_sheet.row_dimensions[int(target_cell[1:])].height = height
            image.width = width * 6.8
            image.height = height * 1.3

            plots_sheet.add_image(image, target_cell)
            pil_image.close()
            os.remove('plot.png')
            col = col + 1

    def save_plots_and_scores(self, col, conf, scores, num_samples=""):
        cdt_cm.log_progress("Saving DAGs - Start")
        workbook = self.get_xl_workbook_for_plots(self.plots_file)
        
        plots_sheet = workbook[self.plots_sheet_name]
        self.save_plots_and_scores_to_xl(plots_sheet, col, scores, 
                                         conf, num_samples)

        if col == 0:
            configs_sheet = workbook[self.config_sheet_name]
            self.save_config_to_xl(configs_sheet, conf)

        workbook.save(self.plots_file)
        workbook.close()
        cdt_cm.log_progress("Saving DAGs - Done")

    def start_training(self, conf):
        dists_file = f"{self.data_folder}/{conf['name']}.dists"
        with open(dists_file, "rb") as file:
            scm_dists = pickle.load(file)
        scm = SCM(scm_dists)
        self.save_plots_and_scores(0, conf, 
                                   {"Original":{"dag": scm.dag}})
        
        self.train_algos_for_config_and_save_results(scm_dists, conf)

