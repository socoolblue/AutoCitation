import os
import re
import time
import pickle
import numpy as np
import PyPDF2
import spacy
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

nlp = spacy.load("en_core_web_sm")

def init_gemini_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)

def split_sentences_with_spacy(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_text_from_pdf(path):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    ref_markers = ["references", "References", "Acknowledgements", "REFERENCES"]
    ref_start = min((text.rfind(m) for m in ref_markers if m in text), default=len(text))
    return text[:ref_start]

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ref_markers = ["references", "References", "Acknowledgements", "REFERENCES"]
    ref_start = min((text.rfind(m) for m in ref_markers if m in text), default=len(text))
    return text[:ref_start]

def process_reference_papers(folder, ext="pdf"):
    documents = []
    for fn in os.listdir(folder):
        if fn.endswith(f".{ext}"):
            path = os.path.join(folder, fn)
            text = extract_text_from_pdf(path) if ext == "pdf" else extract_text_from_txt(path)
            documents.append(Document(page_content=text, metadata={"source": fn}))
    return documents

def preprocess_introduction(intro_text):
    ref_pattern = r"\[(\d+(?:[-\u2013]\d+)?(?:,\s*\d+(?:[-\u2013]\d+)?)*)\]"
    def expand_range(range_str):
        parts = range_str.replace('\u2013', '-').replace(' ', '').split(',')
        numbers = set()
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.update(range(start, end+1))
            else:
                numbers.add(int(part))
        return sorted(numbers)
    
    matches = re.finditer(ref_pattern, intro_text)
    ground_truth = {}
    cleaned_intro = intro_text
    for i, match in enumerate(matches, 1):
        original = match.group(0)
        numbers = expand_range(match.group(1))
        placeholder = f"[ref{i}]"
        ground_truth[placeholder] = numbers
        cleaned_intro = cleaned_intro.replace(original, placeholder, 1)
    return cleaned_intro, ground_truth

def build_vector_store(documents, client, embedding_file="./embeddings.pkl", chunk_size=1000, chunk_overlap=200):
    from google import genai
    try:
        with open(embedding_file, "rb") as f:
            split_documents = pickle.load(f)
        print("Loaded cached embeddings.")
    except:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)
        for i, doc in enumerate(split_documents):
            try:
                time.sleep(0.8)
                result = client.models.embed_content(model="gemini-embedding-exp-03-07", contents=doc.page_content)
                doc.metadata["embedding"] = result.embeddings[0].values
                print(f"Embedded {i+1}/{len(split_documents)}")
            except:
                doc.metadata["embedding"] = [0.0]*768
        with open(embedding_file, "wb") as f:
            pickle.dump(split_documents, f)

    texts = [d.page_content for d in split_documents]
    embeddings = [d.metadata["embedding"] for d in split_documents]
    metadatas = [d.metadata for d in split_documents]
    return FAISS.from_embeddings(list(zip(texts, embeddings)), embedding=None, metadatas=metadatas)

def predict_references_with_RAG(vector_store, cleaned_intro):
    faiss_refs = {}
    for ph in re.findall(r"\[ref\d+\]", cleaned_intro):
        context = extract_context_for_ref(cleaned_intro, ph)
        results = vector_store.similarity_search(context, k=60)
        unique_sources, seen = [], set()
        for r in results:
            src = r.metadata["source"]
            if src not in seen and len(unique_sources) < 5:
                unique_sources.append(src)
                seen.add(src)
        faiss_refs[ph] = unique_sources
    return faiss_refs

def extract_context_for_ref(text, placeholder, before=1, after=0):
    sents = split_sentences_with_spacy(text)
    for i, s in enumerate(sents):
        if placeholder in s:
            start = max(0, i-before)
            end = min(len(sents), i+after+1)
            return ". ".join(sents[start:end])
    return ""

def evaluate_predictions(ground_truth, predicted):
    total_score, report = 0, []
    for ph, true_ids in ground_truth.items():
        predicted_ids = predicted.get(ph, [])
        matched = set(predicted_ids) & set(map(str, true_ids))
        score = len(matched) / len(true_ids) if true_ids else 0
        total_score += score
        report.append({"Placeholder": ph, "Ground Truth": true_ids, "predicted_refs": predicted_ids})
    avg_score = total_score / len(ground_truth) if ground_truth else 0
    return report, avg_score
