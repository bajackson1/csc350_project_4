# Project 4 Submission

Team Members: Jacorian Adom, Aiden Agas, Brooks Jackson, Thomas Morrissey

## Repository

GitHub repository: `https://github.com/bajackson1/csc350_project_4.git`

## Contents
- `run_analysis.py`: entry point for the project
- `src/common.py`: shared imports, constants, and utility helpers
- `src/data_processing.py`: loading, aggregation, preprocessing, and EDA output generation
- `src/modeling.py`: feature matrix creation, model fitting, evaluation, and importance extraction
- `src/reporting.py`: final markdown report generation
- `Project_4_Presentation.pdf`: project presentation slides
- `outputs/`: deliverable paper, markdown report, figures, and tables
- `requirements.txt`: Python package versions used for reproduction

The dataset is intentionally not included in this submission folder.

## How To Reproduce

This project can read the dataset in either of these layouts:

```text
Downloads/
├── Project_4_Adom_Agas_Jackson_Morrissey/
├── loneliness_data/
```

or:

```text
Downloads/
├── Project_4_Adom_Agas_Jackson_Morrissey/
├── Participant_1/
├── Participant_2/
├── ...
```

## Reproduction Steps

1. Place `Project_4_Adom_Agas_Jackson_Morrissey/` in the same parent directory as either `loneliness_data/` or the extracted `Participant_*` folders
2. Open a terminal in `Project_4_Adom_Agas_Jackson_Morrissey/`
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python run_analysis.py
```

## What The Script Generates

Running the script will regenerate:

- `outputs/Project_4_Deliverable.md`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`

The final paper PDF included for submission is:

- `outputs/Project_4_Deliverable_Paper.pdf`

## Required Python Packages

This project uses the package versions listed in `requirements.txt`.
