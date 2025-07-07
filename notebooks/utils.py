import os
import re
import time
import pickle
import random
import PyPDF2
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

nlp = spacy.load("en_core_web_sm")


def split_sentences_with_spacy(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def extract_text_from_pdf(path):
    """Extract main text content from PDF (excluding reference section)."""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)

    ref_markers = ["references", "References", "Acknowledgements", "REFERENCES"]
    ref_start = min((text.rfind(m) for m in ref_markers if m in text), default=len(text))
    return text[:ref_start]


def extract_text_from_txt(path):
    """Extract main text content from .txt file (excluding reference section)."""
    with open(path, "r", encoding='utf-8') as file:
        text = file.read()

    ref_markers = ["references", "References", "Acknowledgements", "REFERENCES"]
    ref_start = min((text.rfind(m) for m in ref_markers if m in text), default=len(text))
    return text[:ref_start]


def process_reference_papers(folder, ext="pdf"):
    """Convert all .pdf or .txt files in a folder into LangChain Documents."""
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(f".{ext}"):
            path = os.path.join(folder, filename)
            print(f"Loading: {filename}")
            text = extract_text_from_pdf(path) if ext == "pdf" else extract_text_from_txt(path)
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def preprocess_introduction(text):
    """Replace numbered citation tags with placeholders and extract mapping."""
    ref_pattern = r"\[(\d+(?:[-\u2013]\d+)?(?:,\s*\d+(?:[-\u2013]\d+)?)*)\]"

    def expand_range(s):
        s = s.replace('\u2013', '-').replace(' ', '')
        refs = set()
        for part in s.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                refs.update(range(start, end + 1))
            else:
                refs.add(int(part))
        return sorted(refs)

    ground_truth = {}
    cleaned_text = text
    for i, match in enumerate(re.finditer(ref_pattern, text), 1):
        placeholder = f"[ref{i}]"
        refs = expand_range(match.group(1))
        ground_truth[placeholder] = refs
        cleaned_text = cleaned_text.replace(match.group(0), placeholder, 1)

    return cleaned_text, ground_truth


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1).flatten(), np.array(vec2).flatten()
    denom = norm(vec1) * norm(vec2)
    return float(dot(vec1, vec2) / denom) if denom != 0 else 0.0


def save_embeddings(documents, filename):
    """Save document list with embeddings to a file."""
    with open(filename, "wb") as f:
        pickle.dump(documents, f)


def load_embeddings(filename):
    """Load document list with embeddings from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def build_vector_store(documents, openai_api_key, embedding_file="./embeddings.pkl",
                       embed_model="text-embedding-3-large", chunk_size=1000, chunk_overlap=200):
    """Build or load a FAISS vector store using OpenAI embeddings."""
    try:
        split_documents = load_embeddings(embedding_file)
        print("Loaded existing embeddings from file.")
    except FileNotFoundError:
        print("No cached embeddings found. Creating new ones...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        embedder = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embed_model,
            chunk_size=chunk_size,
            max_retries=6,
            timeout=10
        )

        for idx, doc in enumerate(split_documents):
            try:
                doc.metadata["embedding"] = embedder.embed_query(doc.page_content)
                print(f"Embedded chunk {idx + 1}/{len(split_documents)}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Embedding error on chunk {idx + 1}: {e}")
                doc.metadata["embedding"] = [0.0] * 1536  # fallback

        save_embeddings(split_documents, embedding_file)

    embedder = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model=embed_model,
        chunk_size=chunk_size,
        max_retries=6,
        timeout=10
    )

    vector_store = FAISS.from_documents(split_documents, embedder)

    print("-------------------------------------------------------------")
    print("Total chunks embedded:", vector_store.index.ntotal)
    print("Unique sources:", len({doc.metadata['source'] for doc in split_documents}))

    return vector_store


def extract_context_for_ref(text, placeholder, sentences_before=1, sentences_after=0):
    """Extract sentences surrounding a citation placeholder."""
    sentences = split_sentences_with_spacy(text)
    for i, sentence in enumerate(sentences):
        if placeholder in sentence:
            start = max(0, i - sentences_before)
            end = min(len(sentences), i + 1 + sentences_after)
            return ". ".join(sentences[start:end])
    print(f"Warning: Placeholder {placeholder} not found.")
    return ""


def predict_references_with_RAG(vector_store, cleaned_intro):
    """Perform similarity search for each placeholder and rank top-k citations."""
    prediction_dict = {}
    placeholders = re.findall(r"\[ref\d+\]", cleaned_intro)

    for ph in placeholders:
        context = extract_context_for_ref(cleaned_intro, ph, 1, 0)
        results = vector_store.similarity_search(context, k=60)

        seen = set()
        top_refs = []
        for doc in results:
            src = doc.metadata["source"]
            if src not in seen and len(top_refs) < 5:
                top_refs.append(src)
                seen.add(src)

        prediction_dict[ph] = top_refs
        print(f"Predicted for {ph}: {top_refs}")

    return prediction_dict


def evaluate_predictions(ground_truth, predictions):
    """Evaluate top-k predictions using simple overlap score."""
    score_sum = 0
    report = []

    for ph, true_refs in ground_truth.items():
        predicted = predictions.get(ph, [])
        overlap = set(map(str, true_refs)) & set(predicted)
        score = len(overlap) / len(true_refs) if true_refs else 0
        score_sum += score
        report.append({
            "Placeholder": ph,
            "Ground Truth": true_refs,
            "Predicted": predicted,
            "Score": score
        })

    average_score = score_sum / len(ground_truth) if ground_truth else 0
    return report, average_score
