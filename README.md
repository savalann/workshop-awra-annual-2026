# NextGen Hydrologic Modeling Workshop: From Forcings to Extreme-Event Evaluation

**Authors:** Savalan Naser Neisary, Parthkumar Modi, Steven Burian
**Contact:** [snaserneisary@crimson.ua.edu](mailto:snaserneisary@crimson.ua.edu)
**Workshop materials on HydroShare:** [HydroShare link to be added]

---

## Abstract

This workshop presents a practical, end-to-end approach to working with water data across the United States, from data collection and preparation to analysis, visualization, and prediction, while highlighting selected tools and resources developed through the Cooperative Institute for Research to Operations in Hydrology (CIROH). Participants will be introduced to Python-based tools for accessing and preprocessing hydrological and meteorological data from multiple sources, including observational datasets and forecasting products such as National Water Model streamflow predictions. The workshop will demonstrate how processed data can be used for multiple applications, including analysis, visualization, and large-scale modeling. Examples will include both conventional and machine learning-based approaches, along with methods for evaluating model performance against observations. Designed for a broad audience including researchers, consultants, and operational practitioners, the session will emphasize practical methods and reusable tools that support real-world water applications. By the end of the workshop, participants will have a clearer understanding of how to collect, prepare, analyze, visualize, and use water data for prediction and decision support, while gaining exposure to CIROH’s broader ecosystem of open-source tools and resources for operational hydrology.

## Learning objectives

1. Retrieve, preprocess, and evaluate AORC and MRMS data against meteorological observations.
2. Run a pretrained LSTM model within NGIAB to predict streamflow for two hydrologically contrasting watersheds using alternative precipitation forcings.
3. Use CSES to visualize and evaluate overall and low-flow streamflow prediction performance, and interpret how performance varies across watersheds, precipitation forcings, and low-flow thresholds.

## Prerequisites

- Basic Python and Jupyter Notebooks.
- Basic hydrology (streamflow, precipitation, rainfall–runoff modeling).

---

## Notebooks

Run the notebooks in order. Each one starts with a short description and ends with a link to the next.

| # | Notebook | What it does |
|---|---|---|
| 1 | `01.forcing_prepration.ipynb` | **NextGen data preparation.** Downloads the CONUS hydrofabric (first run only), subsets it for the gage, builds the AORC forcings, and writes the model configuration (`realization.json`, `troute.yaml`). Includes maps and time series of the forcings. |
| 2 | `02.forcing_evaluation.ipynb` | **Rainfall evaluation.** Compares IMERG and AORC hourly rainfall with rain gauges in and around the basin (2002–2022): gauge quality control, hourly scores (KGE, POD, FAR, CSI), and figures. |
| 3 | `03.model_run.ipynb` | **Model run.** Runs the NextGen LSTM model with t-route routing using the package from notebook 1, and plots the simulated hydrograph against USGS observations. |
| 4 | `04.model_evaluation_teehr.ipynb` | **TEEHR evaluation.** Builds a TEEHR evaluation with USGS observations, the NWM v3.0 retrospective, and the NextGen simulation, then computes performance metrics. |
| 5 | `05.extreme_events_.ipynb` | **Extreme events.** Compares NWM and NextGen runs driven by two rainfall products (AORC and IMERG) against USGS observations over the full record (2002–2022), for drought events, and during the SEED flood and drought events. |

**Study basin:** Sipsey Fork near Grayson, Alabama (USGS gage **02450250**, 243 km²; NWM reach 18578829; NextGen segment 486182).

## Folder structure

```
.
├── 01.forcing_prepration.ipynb
├── 02.forcing_evaluation.ipynb
├── 03.model_run.ipynb
├── 04.model_evaluation_teehr.ipynb
├── 05.extreme_events_.ipynb
├── utils/                     # Python code used by the notebooks (the notebooks only call it)
├── data/                      # ready-made data used by notebooks 2 and 5
└── ngiab_preprocess_output/   # NGIAB inputs and outputs, one folder per gage (created by notebook 1)
```

**`utils/`**

| File | Used by | Purpose |
|---|---|---|
| `workshop_paths.py` | all | Folder locations; sets where NGIAB writes its files |
| `hydrofabric_visualization_utils.py` | 1 | Hydrofabric download and maps |
| `forcings_utils.py` | 1 | Forcing processing and plots |
| `data_prep_plots.py` | 1, 3, 4 | Plots and helper steps (hydrograph, gage lookup, TEEHR registration) |
| `ngiab_utils.py`, `ngen_outputs_utils.py` | 3, 4 | Reading NGIAB inputs and NextGen/t-route outputs |
| `forcing_evaluation_utils.py` | 2 | Rainfall evaluation: quality control, scores, figures |
| `drought_events.py` | 5 | Metrics, drought events and event plots |

**`data/`**

| File | Content |
|---|---|
| `daily_flows.parquet` | Daily streamflow (m³/s), 2002-01-01 to 2023-01-30: USGS observations, NWM v3.0 retrospective, and NextGen runs with AORC, Stage IV and IMERG rainfall (the IMERG run ends 2022-12-30) |
| `drought_thresholds.parquet` | Daily drought thresholds (Q95, Q90, Q80, Q70) per calendar day |
| `sipsey_fork_hourly.csv` | Hourly rainfall at each gauge, with IMERG and AORC at the gauge's grid cell |
| `sipsey_fork_gauges.csv`, `info.json` | Rain gauge list and basin information |
| `basin.geojson`, `flowlines.geojson`, `states.zip` | Basin outline, streams and state boundaries for maps |

---

## Getting started

### Option A: 2i2c JupyterHub (recommended)

1. Log in to the workshop JupyterHub and choose the NGIAB image (`quay.io/awiciroh/ngiab-2i2c:v1.3.1`).
2. Upload or clone this folder into your home directory.
3. Open `01.forcing_prepration.ipynb` and run the cells from top to bottom (**Shift + Enter**).

### Option B: Your own computer with Docker

Anyone with Docker can run the whole workshop. **Nothing else needs to be installed:** Python, all packages, the NGIAB tools, the NextGen model and TEEHR are already inside the Docker image.

**What you need:** Docker, an internet connection, and about 13 GB of free disk space for the image plus about 6 GB for the hydrofabric that notebook 1 downloads. On Apple Silicon Macs the image runs under emulation and is slower.

**Step 1. Download the workshop files and folders.** Download the complete workshop folder from HydroShare ([HydroShare link to be added]) and unzip it. Keep all files and folders together (the five notebooks, `utils/` and `data/`).

**Step 2. Install Docker.** Install [Docker Desktop](https://docs.docker.com/get-started/get-docker/) (Windows or macOS) or Docker Engine (Linux), then start it and wait until it says it is running.

**Step 3. Start the workshop.** Open a terminal (PowerShell on Windows) **inside the workshop folder** and run:

macOS / Linux:

```bash
mkdir -p .ngiab
docker run --rm -it --platform linux/amd64 -p 8888:8888 -v "$PWD":/home/jovyan/workshop -v "$PWD/.ngiab":/home/jovyan/.ngiab quay.io/awiciroh/ngiab-2i2c:v1.3.1 jupyter lab --ip 0.0.0.0 --ServerApp.root_dir=/home/jovyan
```

Windows (PowerShell):

```powershell
mkdir .ngiab
docker run --rm -it --platform linux/amd64 -p 8888:8888 -v "${PWD}:/home/jovyan/workshop" -v "${PWD}/.ngiab:/home/jovyan/.ngiab" quay.io/awiciroh/ngiab-2i2c:v1.3.1 jupyter lab --ip 0.0.0.0 --ServerApp.root_dir=/home/jovyan
```

The first run downloads the image, which takes a while. When it starts, the terminal prints a link such as `http://127.0.0.1:8888/lab?token=...`. Open it in your browser, open the `workshop` folder, and start with notebook 1.

**Notes**
- Keep the terminal open while you work. Press **Ctrl + C** in it to stop. Your files stay in the workshop folder.
- To start again, run the same `docker run` command. The hydrofabric stays in the hidden `.ngiab` folder, so it is downloaded only once.
- If Docker runs out of memory, increase the memory given to Docker in Docker Desktop (Settings → Resources).

---

## Practical notes

- **Restart the kernel** before running a notebook (**Kernel → Restart Kernel and Clear Outputs of All Cells**).
- **The `online` setting (notebooks 1 and 3).** It is set in the first settings cell of the notebook.
  - `online = 'yes'`: run the steps that **download or create the model inputs** (hydrofabric download, subsetting, forcings, model configuration). This needs an internet connection and is required the first time you run the workshop for a gage.
  - `online = 'no'`: **skip** those steps and use the files that are already in `ngiab_preprocess_output/`. Use this only when that folder already has the files for your gage.
- **Hydrofabric download:** the CONUS hydrofabric is about 1.7 GB to download and about 6 GB after unpacking. It is stored in `~/.ngiab` and downloaded only once per machine.
- **Model run time:** a 2-month run takes a few minutes; long runs (decades) can take 20 minutes or more. On 2i2c, use 3 CPUs (`_cpu_count = 3`); more can run out of memory.
- **Forcing period:** the NWM v3.0 retrospective forcing used here ends on **2023-01-31**. Do not set a run end date later than that.
- **TEEHR data (notebook 4):** `data_path = 'offline'` reuses a saved evaluation; `'online'` deletes it and downloads the data again.
- **"Connection refused" in TEEHR** means its Spark engine stopped, usually from low memory. Shut down other kernels, restart the kernel, and run again.

---

## Workshop resources

More material about this course. Add the links when they are ready.

| Resource | What it is for | Link |
|---|---|---|
| Slides | Presentation used during the workshop | [link to be added] |
| Recording | Video of the workshop session | [link to be added] |
| Handout / extra notes | Additional explanations for the notebooks | [link to be added] |
| Workshop page | Announcement and schedule | [link to be added] |

---

## Our research and tools

Other research and tools from our group that you can review after the workshop. Add one row for each.

| Name | What it is | Link |
|---|---|---|
| [paper or tool name] | [one-line description] | [link to be added] |
| [paper or tool name] | [one-line description] | [link to be added] |
| [paper or tool name] | [one-line description] | [link to be added] |

---

## Useful links (external tools and data)

| Resource | What it is for | Link |
|---|---|---|
| CIROH | The Cooperative Institute for Research to Operations in Hydrology: the program behind these tools | [ciroh.ua.edu](https://ciroh.ua.edu) |
| CIROH documentation | Guides for CIROH's open-source tools, infrastructure and training | [docs.ciroh.org](https://docs.ciroh.org) |
| NextGen In A Box (NGIAB) | Containerized NextGen framework: run the model without building it yourself | [NGIAB-CloudInfra](https://github.com/CIROH-UA/NGIAB-CloudInfra) |
| NGIAB data preprocessor | Builds the model inputs (hydrofabric subset, forcings, configuration) for a gage or catchment | [NGIAB_data_preprocess](https://github.com/CIROH-UA/NGIAB_data_preprocess) |
| NGIAB Python wrapper | Runs NextGen from a Jupyter notebook (used in notebook 3) | [ciroh_ngiab_python](https://github.com/fbaig/ciroh_ngiab_python) |
| NextGen framework (ngen) | The NOAA-OWP model framework that runs the models | [NOAA-OWP/ngen](https://github.com/NOAA-OWP/ngen) |
| t-route | Channel routing used by NextGen | [NOAA-OWP/t-route](https://github.com/NOAA-OWP/t-route) |
| Hydrofabric v2.2 data model | Describes the catchments, nexuses and attributes the model uses | [Data model](https://lynker-spatial.s3-us-west-2.amazonaws.com/hydrofabric/v2.2/hfv2.2-data_model.html) |
| TEEHR | Tools for Exploratory Evaluation in Hydrologic Research: metrics and evaluation of simulations | [GitHub](https://github.com/RTIInternational/teehr), [documentation](https://rtiinternational.github.io/teehr/) |
| NWM v3.0 retrospective | National Water Model streamflow and forcings, 1979–2023 | [AWS Open Data](https://registry.opendata.aws/nwm-archive/) |
| AORC | NOAA Analysis of Record for Calibration: hourly gridded forcings | [AWS Open Data](https://registry.opendata.aws/noaa-nws-aorc/) |
| GPM IMERG | NASA satellite precipitation estimates | [gpm.nasa.gov](https://gpm.nasa.gov/data/imerg) |
| MRMS | NOAA Multi-Radar Multi-Sensor precipitation products | [nssl.noaa.gov](https://www.nssl.noaa.gov/projects/mrms/) |
| USGS NWIS | USGS streamflow observations | [waterdata.usgs.gov](https://waterdata.usgs.gov/nwis) |
| dataretrieval (Python) | Downloads USGS data in Python | [GitHub](https://github.com/DOI-USGS/dataretrieval-python) |
| SEED paper | Method and dataset for evaluating models during extreme events | [doi:10.1111/1752-1688.70093](https://doi.org/10.1111/1752-1688.70093) |
| HydroShare | Repository that holds these workshop materials | [hydroshare.org](https://www.hydroshare.org) |
| 2i2c | The JupyterHub service these notebooks run on | [2i2c.org](https://2i2c.org) |
| Docker | Runs the workshop on your own computer | [Get Docker](https://docs.docker.com/get-started/get-docker/) |

---

## How to cite

Please cite this workshop as:

> Naser Neisary, S., Modi, P., & Burian, S. (2026). *NextGen Hydrologic Modeling Workshop: From Forcings to Extreme-Event Evaluation*. HydroShare. https://doi.org/[DOI to be added]

If you use the extreme-event evaluation method, please also cite the SEED paper listed above.

## Acknowledgments

This work was funded by the Cooperative Institute for Research to Operations in Hydrology (CIROH). The workshop builds on the CIROH NextGen/NGIAB workshop materials and runs on the CIROH 2i2c JupyterHub.

The workshop materials are archived on HydroShare. DOI: [HydroShare DOI to be added]
