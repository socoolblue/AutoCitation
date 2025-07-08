import os
import re
import random
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import openai
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import time
from utils import *

openal_api_key_lab="Enter your API key here"


nlp = spacy.load("en_core_web_sm")

def split_sentences_with_spacy(text):
    """Use spaCy to split text into sentences."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def extract_text_from_pdf(pdf_folder):
    """Extract text excluding references from a PDF file."""
    
    with open(pdf_folder, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)
        
        # Find where references begin (common markers)
        ref_markers = ["references","References", "Acknowledgements", "REFERENCES"]
        ref_start = float('inf')
        for marker in ref_markers:
            pos = text.rfind(marker) #find all and pinpoint the last one
            if pos != -1:
                ref_start = min(ref_start, pos)
                
        # Return text up to references section
        if ref_start != float('inf'):
            return text[:ref_start]
        return text

def extract_text_from_txt(txt_folder):
    """Extract text excluding references from a .txt file."""
    with open(txt_folder, "r", encoding='UTF-8') as txt_file:
        lines = txt_file.readlines()
        text = "".join(line for line in lines)
        
        # Find where references begin (common markers)
        ref_markers = ["references","References", "Acknowledgements", "REFERENCES"]
        ref_start = float('inf')
        for marker in ref_markers:
            pos = text.rfind(marker) #find all and pinpoint the last one
            if pos != -1:
                ref_start = min(ref_start, pos)
                
        # Return text up to references section
        if ref_start != float('inf'):
            return text[:ref_start]
        return text


def process_reference_papers(pdf_folder):
    """Process all PDF files in the folder and create documents."""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            print(filename)
            text = extract_text_from_pdf(filepath)
            # Exclude reference lists if needed 
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def process_reference_papers_txt(txt_folder):
    """Process all PDF files in the folder and create documents."""
    documents = []
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(txt_folder, filename)
            print(filename)
            text = extract_text_from_txt(filepath)
            # Exclude reference lists if needed 
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents
    
def preprocess_introduction(intro_text):
 
    ref_pattern = r"\[(\d+(?:[-\u2013]\d+)?(?:,\s*\d+(?:[-\u2013]\d+)?)*)\]"
    
    def expand_range(range_str):
        numbers = set()
    
        parts = range_str.replace('\u2013', '-').replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.update(range(start, end + 1))
            else:
                numbers.add(int(part))
        return sorted(numbers)
    
    matches = re.finditer(ref_pattern, intro_text)
    ground_truth = {}
    ref_count = 1
    
    cleaned_intro = intro_text
    for match in matches:
        original = match.group(0)
        numbers = expand_range(match.group(1))
        placeholder = f"[ref{ref_count}]"
        ground_truth[placeholder] = numbers
        cleaned_intro = cleaned_intro.replace(original, placeholder, 1)
        ref_count += 1
        
    return cleaned_intro, ground_truth

# Utility function for cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    # 벡터 차원 맞추기
    vec1 = np.array(vec1).flatten()  
    vec2 = np.array(vec2).flatten()  
    
   
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0
    return np.dot(vec1, vec2) / norm_product

# Save and Load Embeddings
def save_embeddings(documents, file_name="./embeddings_1000.pkl"):
    """Save embeddings to a file."""
    with open(file_name, "wb") as f:
        pickle.dump(documents, f)

def load_embeddings(file_name="./embeddings_1000.pkl"):
    """Load embeddings from a file."""
    with open(file_name, "rb") as f:
        return pickle.load(f)

# Phase 2: Build Vector Store
def build_vector_store(documents, embedding_file="\\embeddings_1000.pkl", embed_model = 'text-embedding-3-large', chunk_size = 1000, chunk_overlap=200):
    """Create a FAISS vector store from documents with caching for embeddings."""
    try:
        # Try to load embeddings from file
        split_documents = load_embeddings(embedding_file)
        print("Loaded embeddings from file.")
        
    except FileNotFoundError:
        print("Embeddings file not found. Creating embeddings...")
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)
        
        # Generate embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key= openal_api_key_lab, model=embed_model,
            chunk_size=chunk_size, 
            max_retries=6,
            timeout=10
        )
            
        for idx, doc in enumerate(split_documents):
            try:
                doc.metadata["embedding"] = embeddings.embed_query(doc.page_content)
                print(f"Generated embedding for chunk {idx + 1}/{len(split_documents)}")
                time.sleep(0.5)  # Rate limit to avoid API throttling
            except Exception as e:
                print(f"Error embedding document {idx + 1}: {e}")

        # Save embeddings to file for reuse
        save_embeddings(split_documents, embedding_file)
    
    # Create FAISS vector store and ensure embeddings are linked
    vector_store = FAISS.from_documents(split_documents, OpenAIEmbeddings(
            openai_api_key= openal_api_key_lab, model=embed_model,
            chunk_size=chunk_size,  
            max_retries=6,
            timeout=10
        ))
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Total documents in vector store:", vector_store.index.ntotal)
    unique_sources = {doc.metadata["source"] for doc in split_documents}
    print("Unique document sources:", len(unique_sources))
    
    return vector_store

# Phase 3: Reference Prediction
def predict_references_with_RAG(vector_store, cleaned_intro):
    """Predict references for each [ref] in the introduction using RAG"""
    faiss_references_dict = {}

    for placeholder in re.findall(r"\[ref\d+\]", cleaned_intro):
        query_text = extract_context_for_ref(cleaned_intro, placeholder, sentences_before=1, sentences_after=0)



        all_documents = vector_store.similarity_search(query_text, k=60)
        

        unique_references = []
        seen = set()
        
        for doc in all_documents:
            source = doc.metadata["source"]
            if source not in seen and len(unique_references) < 5:
                unique_references.append(source)
                seen.add(source)
        
        faiss_references_dict[placeholder] = unique_references
        print(f"FAISS references for {placeholder}: {unique_references}")

    return faiss_references_dict

# Placeholder context extraction
def extract_context_for_ref(cleaned_intro, placeholder, sentences_before=1, sentences_after=0):
    """Extract a window of sentences around [ref] for focused queries."""
    sentences = split_sentences_with_spacy(cleaned_intro)
    for i, sentence in enumerate(sentences):
        if placeholder in sentence:
            start = max(0, i - sentences_before)
            end = min(len(sentences), i + sentences_after + 1)
            context = '. '.join(sentences[start:end]).strip()
            return context
    print(f"Warning: Placeholder {placeholder} not found in any sentence.")
    return ""


# Phase 6: Scoring Function
def evaluate_predictions(ground_truth, faiss_references_dict):
    """Evaluate predictions against ground truth and calculate the overall score."""
    total_score = 0
    total_placeholders = len(ground_truth)
    detailed_report = []
    
    for placeholder, true_refs in ground_truth.items():
        predicted_refs = faiss_references_dict.get(placeholder, [])

        if len(true_refs) < 5:
            # Ground truth has less than 5 references
            matched_refs = set(predicted_refs) & set(true_refs)
            score = len(matched_refs) / len(true_refs)
        else:
            # Ground truth has 5 or more references
            matched_refs = set(predicted_refs) & set(true_refs)
            score = len(matched_refs) / len(predicted_refs) if predicted_refs else 0

        total_score += score
        
        detailed_report.append({
            "Placeholder": placeholder,
            "Ground Truth": true_refs,
            "predicted_refs": predicted_refs
        })

    # Calculate final score as average across all placeholders
    overall_score = total_score / total_placeholders if total_placeholders > 0 else 0

    return detailed_report, overall_score

