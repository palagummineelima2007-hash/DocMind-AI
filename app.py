import streamlit as st
from pypdf import PdfReader
import io
import re
import math
from collections import Counter

st.set_page_config(
    page_title="DocMind AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocMind AI")
st.subheader("RAG-Based Document Analysis System")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# -----------------------------
# Text similarity functions
# -----------------------------

def tokenize(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def cosine_similarity(text1, text2):

    words1 = tokenize(text1)
    words2 = tokenize(text2)

    count1 = Counter(words1)
    count2 = Counter(words2)

    common_words = set(count1.keys()) & set(count2.keys())

    dot_product = sum(
        count1[word] * count2[word]
        for word in common_words
    )

    magnitude1 = math.sqrt(
        sum(value ** 2 for value in count1.values())
    )

    magnitude2 = math.sqrt(
        sum(value ** 2 for value in count2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# -----------------------------
# PDF processing
# -----------------------------

if uploaded_file is not None:

    st.success("Document uploaded successfully! ✅")

    pdf_data = uploaded_file.read()

    reader = PdfReader(
        io.BytesIO(pdf_data)
    )

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    if text.strip():

        st.write("### 📄 Document Loaded Successfully")

        # -----------------------------
        # Split document into chunks
        # -----------------------------

        words = text.split()

        chunk_size = 100

        chunks = []

        for i in range(0, len(words), chunk_size):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            chunks.append(chunk)

        st.info(
            f"Document divided into {len(chunks)} chunks."
        )

        # -----------------------------
        # Question section
        # -----------------------------

        st.write(
            "### 🤖 Ask Questions About Your Document"
        )

        question = st.text_input(
            "Enter your question"
        )

        if question:

            # Calculate similarity
            scores = []

            for chunk in chunks:

                score = cosine_similarity(
                    question,
                    chunk
                )

                scores.append(score)

            # Get top 3 chunks
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:3]

            st.write(
                "### 🔍 Retrieved Information"
            )

            for index in top_indices:

                st.write(
                    f"**Relevance Score:** "
                    f"{scores[index]:.3f}"
                )

                st.write(chunks[index])

                st.divider()

            st.success(
                "Relevant information retrieved successfully! ✅"
            )

    else:

        st.warning(
            "No readable text found in this PDF."
        )