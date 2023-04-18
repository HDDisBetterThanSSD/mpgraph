# MPGragh: Domain Specific ML Prefetcher for Accelerating Graph Analytics

## Update for SC'23

This repo contains code accompanying the manuscript submitted to SC'23: "Phases, Modalities, Spatial and Temporal Locality: Domain Specific ML Prefetcher for Accelerating Graph Analytics".

## Dependencies

* python: 3.x
* Pytorch: 0.4+
* All dependencies see `environment.yml`

## Artifacts

We download and process graphs from [SNAP](https://snap.stanford.edu/data/). We generate memory access traces using a modified [ChampSim](https://github.com/ChampSim/ChampSim) simulator that supports multi-core OoO simulation. 
