# Resume–Job Fit Classification: Traditional vs. Pretrained Representations

ARTI 404 – Natural Language Processing | Imam Abdulrahman Bin Faisal University | Group 9

**Team:** Bayan Alghamdi, Linah Nashir, Ghala Alsharidah, Lojain Alamri
**Instructor:** Dr. Hayat Alfaghum

## Project overview
The system takes a CV and a job description and predicts how well the candidate fits the job:
**No Fit**, **Potential Fit**, or **Good Fit**. It is intended to support HR employees in screening applicants.

**Challenge:** vocabulary mismatch, where the CV and the job description describe the same skills using different words.

**Experimental question:** Which method better recognises candidate fit when the CV and the job description use different wording?

## Methods
- **Method 1 (traditional):** TF-IDF features, cosine similarity, and Jaccard overlap + Logistic Regression
- **Method 2 (pretrained):** all-MiniLM-L6-v2 sentence embeddings (frozen) + Logistic Regression

## Results (test set)
| Method | Test data | Pairs | Accuracy | Macro F1 |
|---|---|---|---|---|
| Method 1: TF-IDF + LR | Full test | 1108 | 0.644 | 0.645 |
| Method 1: TF-IDF + LR | Low-overlap subset | 294 | 0.650 | 0.611 |
| Method 2: MiniLM + LR | Full test | 1108 | 0.625 | 0.627 |
| Method 2: MiniLM + LR | Low-overlap subset | 294 | 0.599 | 0.557 |

Full results, confusion matrices, and error examples are saved in the `results/` folder.

## How to run
The project was developed as a **Jupyter notebook in VS Code** with Python 3.13.

1. Clone the repository:
   ```
   git clone https://github.com/bayanAlghamdi/resume-job-fit-classification.git
   cd resume-job-fit-classification
   ```
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Open `resume_job_fit.ipynb` and select the Python environment as the kernel.
4. Click **Run All**. The dataset and the pretrained model are downloaded automatically on the first run (about 90 MB for the model).
5. To test a new CV and job description, edit `CV_TEXT` and `JOB_TEXT` in the **Try it** section and run that cell.

All random steps use a fixed seed (42), so the results are reproducible.

## Repository structure
```
resume_job_fit.ipynb   Main notebook (Setup, Data, Method 1, Method 2, Results, Try it, Original Examples, Sources)
requirements.txt       Package versions
results/               Saved results tables, confusion matrices, and error examples
```

## Data and model sources
- Dataset: [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
- Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)