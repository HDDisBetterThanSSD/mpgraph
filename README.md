# MPGragh: Domain Specific ML Prefetcher for Accelerating Graph Analytics

## Update for SC'23

This repo contains code accompanying the manuscript submitted to SC'23: "Phases, Modalities, Spatial and Temporal Locality: Domain Specific ML Prefetcher for Accelerating Graph Analytics".

## Platform

* Ubuntu 20.04.1 LTS. 

## Dependencies

* g++ 9.4.0
* python 3.8
* Pytorch: 1.10.2
* All dependencies see `environment.yml`

## Artifacts Description

1. Trace_generation. We download and process graphs from [SNAP](https://snap.stanford.edu/data/) and use [Intel Pin tool](https://software.intel.com/sites/landingpage/pintool/docs/98484/Pin/html/index.html) to generate the graph processing instruction trace for a specific algorithm and dataset.

2. ChampSim_MPG. We generate memory access traces using a modified [ChampSim](https://github.com/Quangmire/ChampSim) simulator that supports multi-core OoO simulation. This modified ChampSim also supports calling models trained using pytorch to realize ML-based prefetcher simulation.

3. DS Models. We design domain specific (DS) ML models for MPGraph, including

   * Phase detection models: using Soft-KSWIN and Soft-DT for graph processing phase detection under the scenarios of label accessible and label unaccessible, respectively.

   * Spatial delta predictor: predict memory accesses within a spatial range (page) using attention-based network using multi-modality attention fusion (AMMA).
   * Temporal page predictor: predict temporal future pages using AMMA.

4. CSTP controller. The prefetching controller using a chain-spatial-temporal prefetching (CSTP) strategy. It makes use of the spatial delta and temporal page predictions with the help of a Page Base Offset Table (PBOT).



## Run Artifacts

As an example, we use the framework [GPOP](https://github.com/souravpati/GPOP), algorithm "connected components", and dataset [soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html) to show the commands for running the artifacts. It can be extended to other frameworks, algorithms, and datasets by simply changing the command arguments. 

Assume the root directory is `/root/mpgraph`.

### 1. Generate Instruction Trace

1.  Download graph data: `/1_download_graph.sh <graph download link>`. For example:
    * `cd /root/mpgraph/1_Trace_generation/scripts`
    * `./1_download_graph.sh https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz`
    * Graph data is downloaded to `/root/mpgraph/data/raw/`
2.  Generate CSR file for GPOP usage: `./2_gen_gpop_csr.sh <input graph file name> <output csr file name>`. For example:
    * `./2_gen_gpop_csr.sh soc-LiveJournal1.txt soclj`
    * CSR file is stored in `/root/mpgraph/data/csr/`
3.  Generate instruction trace compatible to ChampSim using Pin Tool: `./3_gen_trace_gpop.sh <algorithm> <graph>`. For example:
    * `./3_gen_trace_gpop.sh cc soclj`
4.  Compress the trace for ChampSim usage: `gzip <trace path>`
    * `gzip ../../data/GPOP/ChampSimTraces/cc.soclj*`
    * The instruction trace is stored in `/root/mpgraph/data/GPOP/ChampSimTraces/`.

### 2. Generate Memory Access Trace

1. Compile trace generator for 4 cores: 
   * `cd /root/mpgraph/2_ChampSim_MPG`
   * `./build_champsim.sh bimodal no no no trace lru 4`

2. Generate memory access trace in the shared LLC: `./run_4core_gen_trace.sh <binary> <skip instructions> <simulation instructions> <instruction trace name>`. For example:
   * `cd 0_my_scripts/`
   * `./run_4core_gen_trace.sh bimodal-no-no-no-trace-lru-4core 0 10 cc.soclj.trace`

3. Process and clean the trace for model training: `python process_load_trace.py <algorithm> <graph>`. For example:
   * `python ./process_load_trace.py cc soclj`
   * The memory access trace is stored in `/root/mpgraph/data/GPOP/LoadTraces`.

### 3. Phase Transition Detection

1. Soft KSWIN using unsupervised learning: `python ./run_ks.py <algorithm> <graph>`. For example:

   * `cd /root/mpgraph/3_DS_Models/1_Phase_detection/1_soft_KSWIN`
   * `python ./run_ks.py cc soclj`
   * Result is stored in `/root/mpgraph/res/1_Phase_detection/1_soft_KSWIN`

2. Soft-DT using supervised learning: `python ./run_dt.py <algorithm> <graph>`. For example:

   * `cd /root/mpgraph/3_DS_Models/1_Phase_detection/2_soft_DT`

   * `python ./run_dt.py cc soclj`
   * Result is stored in `/root/mpgraph/res/1_Phase_detection/2_soft_DT`

### 4. Spatial Delta Prediction

1. Preprocessing the trace for model input and labels: 
   * `cd /root/mpgraph/3_DS_Models/2_Spatial_delta_predictor/0_gen_pd`
   * `python run.py <algorithm> <graph> <million_instructions>`. For example:
     * `python run.py cc soclj 2`
       * In default 1M instructions for training and 1M instructions for testing 
   * Result is stored in `/root/mpgraph/data/GPOP/LoadTraces_pd`
2. Phase-specific model training and evaluation: 
   * `cd /root/mpgraph/3_DS_Models/2_Spatial_delta_predictor/SP_AMMA`
   * `python run.py algorithm> <graph> <million_instructions> <phase>`. For example:
     * `python run.py cc soclj 2 scatter`
     * `python run.py cc soclj 2 gather`
   * Model and training hyper-parameters are set in  `config.py` file.
   * Trained models, evaluation results, and training logs are stored in `/root/mpgraph/res/2_Spatial_delta_predictor` as `.pth`, `.csv`, and `.log` files, respectively.

### 5. Temporal Page Prediction

1. Phase specific model training and evaluation: 
   * `cd /root/mpgraph/3_DS_Models/3_Temporal_page_predictor/TP_AMMA`
   * `python run.py algorithm> <graph> <million_instructions> <phase>`. For example:
     * `python run.py cc soclj 2 scatter`
     * `python run.py cc soclj 2 gather`
   * Model and training hyper-parameters are set in  `config.py` file.
   * Trained models, evaluation results, and training logs are stored in `/root/mpgraph/res/3_Temporal_page_predictor` as `.pth`, `.csv`, and `.log` files, respectively.

### 6. Prefetching Simulation with CSTP Controller

1. Compile the simulator:

   * `cd /root/mpgraph/2_ChampSim_MPG`

   * For no prefetcher:
     * `./build_champsim.sh bimodal no no no no lru 4`
   * For the MPGraph prefetcher, which calls the DS ML models through the CSTP controller: 
     * `./build_champsim.sh bimodal no no no py_pref lru 4`

2. Run simulation: `./run_4core_my.sh <binary> <skip million instructions> <simulate million instructions> <mix instructions> <phase> <instruction trace name>`. For example:

   * * `cd ./0_my_scripts/`
     * For no prefetcher:
       *  `./run_4core_my.sh bimodal-no-no-no-no-lru-4core 1 2 0 scatter cc.soclj.trace`
       *  `./run_4core_my.sh bimodal-no-no-no-no-lru-4core 1 2 0 gather cc.soclj.trace`
     * For MPGraph: 
       * `./run_4core_my.sh bimodal-no-no-no-py_pref-lru-4core 1 2 0 scatter cc.soclj.trace`
       * `./run_4core_my.sh bimodal-no-no-no-py_pref-lru-4core 1 2 0 gather cc.soclj.trace`

3. The simulation reports are stored in: `/root/mpgraph/res/4_Simulation`

 ## DOI

 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7844102.svg)](https://doi.org/10.5281/zenodo.7844102)
