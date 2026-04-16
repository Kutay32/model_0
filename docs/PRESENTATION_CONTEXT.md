# Presentation context (curated web references)

Use these when defending the project or writing slides: they situate **breast mammography AI**, **CBIS-DDSM-style data**, **segmentation + classification pipelines**, and **clinical translation** next to your CLIP + U-Net implementation.

## Dataset and benchmarks

1. **CBIS-DDSM (TCIA)** — Curated breast mammography subset used widely for detection and segmentation research.  
   https://www.cancerimagingarchive.net/collection/cbis-ddsm/

2. **CBIS-DDSM dataset description (PMC)** — “A curated mammography data set for use in computer-aided detection and diagnosis research.”  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5735920/

3. **Kaggle mirror (matches `mammography/download_cbis.py` default)** — Convenient packaged layout for local training.  
   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

## Clinical and methods context

4. **Deep learning surveys on mammography** — Search for recent open-access reviews on “deep learning mammogram classification PMC” to cite current screening and CAD challenges.

5. **Multi-site evaluation** — When discussing federated learning from your PDF, connect to **domain shift** across scanners and sites (common in mammography AI validation).

## How to phrase your demo

- Position the web UI as a **decision-support prototype**, not a diagnostic device.
- Emphasize **data locality**: training scripts and the API run on local weights; no cloud inference is required for the demo.
- Connect **federated learning** (from your PDF) to **federated weights**: exported `best_clip_classifier.pth` and `best_unetplusplus_model.pth` are the kind of artifacts a FedAvg server could aggregate without seeing raw images.

## Your stack in one sentence

“Fine-tuned **CLIP** for three-way lesion triage and a **ResNet-34 UNet++** with BCE+Dice-style training for mask estimation on **mammogram patches**, exposed through a **FastAPI** backend and a static **browser UI** for presentation.”
