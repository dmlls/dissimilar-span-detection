import sys
from enum import Enum
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.embedding_sentence_diff import EmbeddingSentenceDiff
from evaluation.token_classification_sentence_diff import (
    TokenClassificationSentenceDiff,
)

# TODO: Update.
FINETUNED_MODEL_PATH: str = (
    "../../span-evaluation-results/04_TokenClassificationSentenceDiff/transformer_distilbert_distilbert-base-uncased_20240909-203156/01/trained_model/checkpoint-500"
)
BASE_EMBEDDING_SENTENCE_DIFF_MODEL = "all-MiniLM-L6-v2"


class Method(str, Enum):
    TOKEN_CLASSIFICATION_SENTENCE_DIFF = "TokenClassificationSentenceDiff"
    EMBEDDING_SENTENCE_DIFF = "EmbeddingSentenceDiff"


def main():
    st.set_page_config(page_title="SentenceDiff - Demo", page_icon="🔀")
    st.header("SentenceDiff - Demo App")
    st.markdown("---")
    # Modify CSS to:
    # - Prevent full screen on images. (See
    #   https://discuss.streamlit.io/t/hide-fullscreen-option-when-displaying-images-using-st-image/19792).
    # - Remove "Running" icon.
    # - Remove menu button.
    # - Draw a red box around annotations.
    hide_img_fs = """
    <style>
    button[title="View fullscreen"]{ visibility: hidden; }
    .stDeployButton { visibility: hidden; }
    [data-testid="stStatusWidget"] { visibility: hidden; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    font {
        width: 300px;
        border-radius: 5px;
        background: #fd0a0a3d;
        margin: 2px;
        border: solid #a20000 1px;
        padding: 1px 5px;
    }
    </style>
    """
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    method = st.radio(
        "Method to use",
        options=[method.value for method in Method],
        help=(
            f"{Method.TOKEN_CLASSIFICATION_SENTENCE_DIFF.value} uses"
            " a model specifically fine-tuned on the task of semantic dissimilarity."
            f"\n{Method.EMBEDDING_SENTENCE_DIFF.value} is an unsupervised method"
            " powered by a Sentence Transformer model."
        ),
    )
    if "method" not in st.session_state or st.session_state.method != method:
        with st.spinner("Loading model"):
            st.session_state.method = method
            if method == Method.TOKEN_CLASSIFICATION_SENTENCE_DIFF.value:
                st.session_state.model = TokenClassificationSentenceDiff(
                    model=FINETUNED_MODEL_PATH
                )
            elif method == Method.EMBEDDING_SENTENCE_DIFF.value:
                st.session_state.model = EmbeddingSentenceDiff(
                    model=BASE_EMBEDDING_SENTENCE_DIFF_MODEL
                )

    st.divider()

    premise = st.text_input(
        "Input the premise",
        value="The European Union is currently undergoing the installation of new IT systems.",
    )
    hypothesis = st.text_input(
        "Input the hypothesis",
        value="The EU is currently carrying out the removal of previously installed Information Technology systems.",
    )

    if st.session_state.method == Method.EMBEDDING_SENTENCE_DIFF.value:
        threshold = st.number_input(
            "Threshold",
            value=0.007,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%0.3f",
            help="Adjust this value to obtain more precise annotations.",
        )
    else:
        threshold = None

    if st.button("Annotate"):
        if premise and hypothesis:
            with st.spinner("Processing"):
                st.divider()
                annotated_hypothesis = st.session_state.model.annotate_diff(
                    premise, hypothesis, token_diff_threshold=threshold
                )
                annotated_hypothesis = annotated_hypothesis.replace(
                    "{{", "<font>"
                ).replace("}}", "</font>")
                st.markdown("**Annotated hypothesis:**")
                st.html(annotated_hypothesis)
        else:
            st.error("Please, specify the premise and hypothesis.")


if __name__ == "__main__":
    main()
