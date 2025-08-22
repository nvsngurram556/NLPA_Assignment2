# libraries required for Data Extraction and Processing
import re
from collections import Counter
# libararies for text processing and embeddings
import uuid
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# libraries for vector store and retriever
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# libraries for text generation
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#CLI interface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time



def clean_text(raw_text):
    # Split text into lines
    if isinstance(raw_text, list):
        raw_text = "\n".join(raw_text)
    lines = raw_text.splitlines()

    # Count line frequency to identify repeated headers/footers
    line_counts = Counter(lines)

    cleaned_lines = []
    seen_lines = set()
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        # Skip if line appears too often (likely header/footer)
        if line_counts[line] > 3:  # adjust threshold
            continue
        # Skip page numbers
        if re.match(r'^\s*(page\s*\d+|\d+\s*of\s*\d+|\d+)\s*$', line, re.IGNORECASE):
            continue
        # Skip lines that look like headers with patterns like "--- filename | Page number ---"
        if re.match(r'^---.*Page\s*\d+.*---$', line, re.IGNORECASE):
            continue
        # Skip duplicate lines
        line_stripped = line.strip()
        if line_stripped in seen_lines:
            continue
        seen_lines.add(line_stripped)
        cleaned_lines.extend(line_stripped.split('. '))  # Split long lines into sentences

    # Join lines and normalize spaces
    text = " ".join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_sections(text):
    # Define section headings to look for
    sections = ["Income Statement", "Balance Sheet", "Cash Flow Statement", "Notes"]
    # Create a regex pattern to split on these headings
    pattern = re.compile(r'(' + '|'.join([re.escape(section) for section in sections]) + r')', re.IGNORECASE)
    # Split text by the headings, keeping the headings
    parts = pattern.split(text)
    segmented = {}
    current_section = None
    for part in parts:
        part_strip = part.strip()
        # Check if this part is a section heading
        if any(part_strip.lower() == sec.lower() for sec in sections):
            current_section = part_strip
            segmented[current_section] = ""
        elif current_section:
            segmented[current_section] += part_strip + " "
    # Strip trailing spaces from each section text
    for key in segmented:
        segmented[key] = segmented[key].strip()
    return segmented

def split_into_chunks(data, chunk_size):
    chunks = []
    for text in data:  # text is already a string
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

def simple_sent_tokenize(text):
    # Split on periods, question marks, exclamation marks, keeping punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty strings
    return [s.strip() for s in sentences if s.strip()]

def create_faiss_vector_store(chunks, index_path):
    if not chunks:
        raise ValueError("No text chunks provided to FAISS indexer.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store

def create_bm25_retriever(chunks):
    bm25_retriever = BM25Retriever.from_texts(chunks)
    return bm25_retriever

def process_text(data):
    # If already a list of chunks, skip preprocessing
    if isinstance(data, list):
        chunks = data
    else:
        preprocessed_data = clean_text(data)
        segmented_data = segment_sections(preprocessed_data)
        # Handle case where segmented_data is a dictionary
        if isinstance(segmented_data, dict):
            segmented_text = "\n\n".join(segmented_data.values())
        else:
            segmented_text = segmented_data

        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        chunks = text_splitter.split_text(segmented_text)

    if not chunks:
        raise ValueError("No text chunks generated during processing.")

    index_path = "data/processed/faiss_index"
    faiss_store = create_faiss_vector_store(chunks, index_path)
    bm25_retriever = create_bm25_retriever(chunks)
    return faiss_store, bm25_retriever


def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    tokens = query.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5):
    processed_query = preprocess_query(query)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embeddings.embed_query(processed_query)

    # Retrieve from FAISS
    faiss_results = faiss_store.similarity_search_by_vector(query_embedding, k=top_n)

    # Retrieve from BM25
    bm25_results = bm25_retriever.get_relevant_documents(processed_query)[:top_n]

    # Combine results by union (simple concatenation)
    combined_results = faiss_results + bm25_results
    return combined_results

def rerank_with_cross_encoder(query, retrieved_docs, top_k=5):
    pairs = []
    for doc in retrieved_docs:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        pairs.append((query, content))
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:top_k]]
    return top_docs

def validate_query(query):
    """
    Checks if the query is valid (not harmful or irrelevant).
    Returns True if valid, False if rejected.
    """
    harmful_keywords = ["hack", "attack", "illegal"]
    irrelevant_keywords = ["joke", "funny"]
    q = query.lower()
    for word in harmful_keywords + irrelevant_keywords:
        if word in q:
            return False
    return True


def validate_response(response):
    """
    Checks for hallucinations or non-factual patterns in the response.
    Returns (is_valid, issues) where issues is a list of detected problems.
    """
    suspicious_phrases = [
        "i don't know", "maybe", "guess"
    ]
    issues = []
    r = response.lower()
    for phrase in suspicious_phrases:
        if phrase in r:
            issues.append(f"Suspicious phrase: '{phrase}'")
    # Check for numbers without context (simple heuristic: lone numbers)
    numbers = re.findall(r'\b\d+(\.\d+)?\b', response)
    # If numbers exist but common context words not found
    context_keywords = ["dollar", "usd", "percent", "percentage", "year", "month", "eps", "revenue", "profit", "loss"]
    for num in numbers:
        has_context = any(ck in r for ck in context_keywords)
        if not has_context:
            issues.append(f"Number '{num}' may lack context")
            break
    is_valid = len(issues) == 0
    return (is_valid, issues)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=generator)

def generate_response(query, top_docs, max_tokens=1500):
    # Concatenate passages up to max_tokens limit (approximate by character length)
    max_context_length = 1500  # Approximate max characters to keep within model context
    concatenated_passages = ""
    for doc in top_docs[:2]:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        if len(concatenated_passages) + len(content) + 1 > max_context_length:
            break
        concatenated_passages += content + "\n"

    prompt = f"Answer the question using the context.\nContext: {concatenated_passages}\nQuestion: {query}\nAnswer:"
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"Error: {e}. Please pull the model using 'ollama pull llama2:7b' and try again.")
        response = ""
    return response


if __name__ == "__main__":
    print("\n--- RAG Pipeline Started ---\n")

    with open("data/raw/irfc_combined_text.txt", "r", encoding="utf-8") as f:
        data_preprocess_code = f.read()
    prepreocessed_data = clean_text(data_preprocess_code)
    segmented_data = segment_sections(prepreocessed_data)

    for section, content in segmented_data.items():
        print(f"Section: {section}, Length: {len(content)}")
    
    chunks_100 = split_into_chunks(segmented_data, 100)
    chunks_400 = split_into_chunks(segmented_data, 400)
    print(f"Number of 100-word chunks: {len(chunks_100)}")
    if chunks_100:
        print("First 100-word chunk example:", chunks_100[0])

    print(f"Number of 400-word chunks: {len(chunks_400)}")
    if chunks_400:
        print("First 400-word chunk example:", chunks_400[0])

    faiss_store, bm25_retriever = process_text(chunks_400)

    #sample_query = "What was EPS in FY2024-25 vs. FY2023-24?"
    #results = hybrid_retrieve(sample_query, faiss_store, bm25_retriever, top_n=5)

    #print(f"Hybrid Retrieval Results for query '{sample_query}':")
    #for idx, res in enumerate(results):
    #    print(f"Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}")

    #reranked_results = rerank_with_cross_encoder(sample_query, results, top_k=5)
    #print("\nReranked Results:")
    #for idx, res in enumerate(reranked_results):
    #    print(f"\nReranked Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}\n")
    
    #generated_answer = generate_response(sample_query, reranked_results, max_tokens=1500)
    #print(f"Generated Answer:\n{generated_answer}\n")

    #print("Validating Query and Response...")
    #is_valid, issues = validate_response(generated_answer)
    #if is_valid:
    #    print("Response is valid.")
    #else:
    #    print("Response is invalid. Issues found:")
    #    for issue in issues:
    #        print(f" - {issue}")

    #print("\n--- RAG Pipeline Completed ---\n")
    

    #CLI interface using Streamlit
    st.title("RAG System for Financial Document Q&A")
    mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"])
    query = st.text_input("Enter your question")
    if query:
        start_time = time.time()
        if mode == "RAG":
            top_chunks = hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5)
            reranked_results = rerank_with_cross_encoder(query, top_chunks)
            answer = generate_response(query, reranked_results)
            validation_passed, validation_issues = validate_response(answer)
            if not validation_passed:
                st.warning("Response validation issues detected:")
                for issue in validation_issues:
                    st.write(f" - {issue}")
            confidence_score = sum([res.metadata.get('similarity', 0) for res in reranked_results]) / len(reranked_results) if reranked_results else 0
        else:
            st.info("Fine-tuned model response generation not implemented yet.")
            answer = ""
            confidence_score = 0
        elapsed_time = time.time() - start_time
        
        if answer:
            st.success(f"Answer: {answer}")
            st.write(f"Confidence Score: {confidence_score:.2f}")
            st.write(f"Method Used: {mode}")
            st.write(f"Response Time: {elapsed_time:.2f} seconds")