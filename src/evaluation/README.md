## Running the evaluation

### Examples

#### Baselines
```
python run_evaluation.py baseline no-sentence-diff ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

```
python run_evaluation.py baseline naive-sentence-diff ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

#### SHAPSentenceDiff
```
python run_evaluation.py shap all-MiniLM-L6-v2 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

#### LIMESentenceDiff
```
python run_evaluation.py lime all-MiniLM-L6-v2 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

#### EmbeddingSentenceDiff
```
python run_evaluation.py sentence-transformer all-MiniLM-L6-v2 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

#### LLMSentenceDiff
```
python run_evaluation.py openai gpt-4-turbo-2024-04-09 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

```
python run_evaluation.py mistral mistral-medium ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv

```

#### TokenClassificationSentenceDiff

```
python run_evaluation.py transformer distilroberta-base-sentence-diff ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
```

To see all available options, you can run: 

```
python run_evaluation.py --help
```
