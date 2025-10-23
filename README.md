### Installing the dependencies

#### Option 1
You can install the dependencies via `pip`:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Option 2

Another option is to use [`pipenv`](https://pipenv.pypa.io/en/latest/index.html), which we use to manage the project dependencies. It can be installed with:

```
pip install pipenv
```

Then, to install all depenendencies and activate the created virtual environment, run:

```
pipenv sync && pipenv shell
```

### Project Structure

We include below the entire project structure. Bear in mind that relevant directories contain further `README.md` files with instructions on how to run the code.

```
.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── requirements.txt
├── span-similarity-dataset                       # SDS and manual annotations
│   ├── annotations
│   ├── README.md
│   └── span_similarity_dataset_v1.0.0.tsv
└── src
    ├── annotation                                # Inter-annotator agreement code
    ├── app                                       # Streamlit app showcasing SentenceDiff
    │   ├── __init__.py
    │   ├── main.py
    │   └── README.md
    ├── dataset                                   # Dataset-related code
    │   ├── chat-gpt-prompt.txt
    │   ├── data_loading.py
    │   ├── get_annotated_pairs.py
    │   ├── get_dataset_statistics.py
    │   ├── __init__.py
    │   ├── models.py
    │   ├── README.md
    │   └── validate_dataset.py
    ├── evaluation                                # Evaluation-related code
    │   ├── baselines_sentence_diff.py
    │   ├── base_sentence_diff.py
    │   ├── config.py
    │   ├── embedding_sentence_diff.py
    │   ├── embeddings.py
    │   ├── env-example
    │   ├── evaluation.py
    │   ├── __init__.py
    │   ├── lime_sentence_diff.py
    │   ├── llm_sentence_diff.py
    │   ├── models.py
    │   ├── README.md
    │   ├── run_evaluation.py                     # Entry point to run evaluation
    │   ├── shap_sentence_diff.py
    │   └── token_classification_sentence_diff.py
    ├── ists                                      # Data and tools for the SemEval-2016 Task 2
    │   ├── ists_datasets
    │   ├── ists_ssd.tsv
    │   └── ists_to_ssd.py
    ├── paraphrase                                # DSD applied to paraphrase detection
    │   ├── PAWS-Wiki-labeled-final
    │   ├── evaluation.py
    │   └── run_evaluation.py
    └── training                                  # Fine-tuning of supervised models
        └── token_classification
```
