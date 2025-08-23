# libraries required for Data Extraction and Processing
import re
from collections import Counter
# libraries for text processing and embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# libraries for vector store and retriever
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
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
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time

#3. Fine-Tuned Model System Implementation

import json
import evaluate
import torch
import pandas
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, pipeline


#2. Retrieval-Augmented Generation (RAG) System Implementation

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

def validate_input_question(question: str) -> bool:
    """
    Validates the input question for the QA system.
    Returns False if the question is empty or longer than 300 characters.
    Otherwise returns True.
    Logs a warning if the question is skipped due to invalid length.
    """
    if not question or len(question) > 300:
        logging.warning("Question skipped due to invalid length (empty or > 300 characters).")
        return False
    return True

def response_val(question: str, max_length: int = 100):
    if not validate_input_question(question):
        logging.warning("Invalid question blocked by guardrail.")
        return "Your question is invalid (empty or too long). Please rephrase."

    prompt = f"Question: {question}\nAnswer:"
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    print(f"\nGenerated output: {output}")
    
    if output and isinstance(output, list) and "generated_text" in output[0]:
        generated_text = output[0]["generated_text"]
        logging.info(f"Generated output: {generated_text}")
        return generated_text
    else:
        logging.error("No output generated by the model.")
        return "Sorry, the model did not produce an answer."


def model_creation():
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    llm = HuggingFacePipeline(pipeline=generator)
    return llm, tokenizer, model, generator


def generate_response(llm, query, top_docs, max_tokens=1500):
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


def get_confidence_rag(rernk_results):
    # Example: average similarity score if available, else 1.0
    scores = [res.metadata.get('similarity', 1.0) for res in rernk_results if hasattr(res, 'metadata')]
    return sum(scores) / len(scores) if scores else 1.0

def correctness(model_answer, ground_truth):
    # Simple string match, can be improved
    return "Y" if ground_truth.lower() in model_answer.lower() else "N"


# 3. Fine-Tuned Model System Implementation
# 3.1 Q/A Dataset Preparation
# * Use the same ~50 Q/A pairs as for RAG but convert into a fine-tuning dataset format.

def convert_qa_to_squad_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all Q: ... A: ... pairs
    pattern = re.compile(r'Q: (.*?)\nA: (.*?)(?=\nQ: |\Z)', re.DOTALL)
    pairs = pattern.findall(content)

    squad_format = []
    for question, answer in pairs:
        answer = answer.strip().replace('\n', ' ')
        item = {
            'context': answer,
            'question': question.strip(),
            'answers': {
                'text': [answer],
                'answer_start': [0]  # since context = answer, start is 0
            }
        }
        squad_format.append(item)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_format, f, ensure_ascii=False, indent=2)
    return squad_format


def convert_txt_to_json(input_path, output_path):
    qa_list = []
    question = None
    answer = None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:"):
                answer = line[2:].strip()
                if question is not None and answer is not None:
                    qa_list.append({
                        "question": question,
                        "answer": answer
                    })
                    question = None
                    answer = None

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(qa_list, out_f, ensure_ascii=False, indent=2)

    print(f"Converted {len(qa_list)} Q/A pairs to {output_path}")


# 3.2 Model Selection
# * Choose a small open-source language model suitable for fine-tuning:
#     * Examples: DistilBERT, MiniLM, GPT-2 Small/Medium, Llama-2 7B, Falcon 7B, Mistral 7B.
# * Ensure no use of closed or proprietary APIs.

#llm, tokenizer, model, generator = model_creation()

# 3.3 Baseline Benchmarking (Pre-Fine-Tuning)
# * Evaluate the pre-trained base model on at least 10 test questions.
# * Record accuracy, confidence (if available), and inference speed.

def pre_train_model(model, tokenizer, test_samples):
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    baseline_results = evaluate_model(
    qa_pipeline,
    [{"question": sample["question"], "context": sample["context"]} for sample in test_samples]
    )
    filtered_baseline_results = response_val(baseline_results, max_length=150)
    for res in filtered_baseline_results:
        if isinstance(res, dict):
            print(f"Question: {res['question']}\nAnswer: {res['answer']}\nInference Time: {res['inference_time']:.2f} seconds\n")
        else:
            print(f"Answer: {res}\n")

# def preprocess_function(examples):
#     inputs = [f"{ex['instruction']}\n{tokenizer.bos_token}" for ex in examples]
#     targets = [f"{ex['output']}{tokenizer.eos_token}" for ex in examples]
#     model_inputs = tokenizer(inputs, truncation=True, padding="longest")
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, truncation=True, padding="longest")
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

def QandA_model_train(tokenizer, model, dataset):
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["question", "answer"]
    )
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./sft_model",
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_dir="./logs",
        remove_unused_columns=False,
        report_to="none",
        bf16=True if torch.cuda.is_available() else False,
    )
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator
    )
    # Train the model
    trainer.train()
    # Save the fine-tuned model
    trainer.save_model("./sft_model")
    tokenizer.save_pretrained("./sft_model") 



def preprocess_function(examples):
    questions = [str(q) for q in examples["question"]]
        # SQuAD-style: answers is a list of dicts with "text" key (list of strings)
    answers = [str(a["text"][0]) if isinstance(a, dict) and "text" in a and len(a["text"]) > 0 else "" for a in examples["answer"]]

    inputs = [q for q in questions]
    targets = [a for a in answers]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs 

def evaluate_model(qa_pipeline, test_questions):
    results = []
    for question in test_questions:
        start_time = time.time()
        if isinstance(question, dict):
            prompt = f"Context: {question.get('context', '')}\nQuestion: {question.get('question', '')}\nAnswer:"
        else:
            prompt = str(question)
        response = qa_pipeline(prompt)
        end_time = time.time()
        if isinstance(response, list) and len(response) > 0:
            answer_text = response[0].get("generated_text", "")
        else:
            answer_text = str(response)
        results.append({
            "question": question,
            "answer": answer_text,
            "inference_time": end_time - start_time
        })
    return results

def route_query(query):
    # Simple routing logic: you can use keyword matching, classifier, or embedding similarity
    finance_keywords = ["revenue", "profit", "dividend", "net worth", "debt", "equity", "EPS"]
    if any(word in query.lower() for word in finance_keywords):
        return "finance"
    else:
        return "general"

def multi_expert_answer(query, expert_models,max_length=100):
    expert_key = route_query(query)
    expert = expert_models[expert_key]
    prompt = f"Question: {query}\n Answer: {query}\nAnswer:"
    result = expert(prompt, max_new_tokens=max_length)
    return result[0]['generated_text'].split("Answer:")[-1].strip()

def expert_model_fine_tune(dataset, model, tokenizer, query):
    expert_models = {
        "finance": pipeline("text-generation",
                            model=AutoModelForCausalLM.from_pretrained("gpt2"),
                            tokenizer=AutoTokenizer.from_pretrained("gpt2")),
    }
    test_query = query
    answer = multi_expert_answer(test_query, expert_models)
    print(f"Expert selected: {route_query(test_query)}")
    print(f"Answer: {answer}")
    return answer


# Fine-Tuning
# * Fine-tune the selected model on your Q/A dataset.
# * Log all hyperparameters:
#     * Learning rate, batch size, number of epochs, compute setup (CPU/GPU).
# * Use efficient techniques as assigned (see next).

@st.cache_resource
def load_finetuned_model(dataset):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./sft_model")
        model = AutoModelForCausalLM.from_pretrained("./sft_model")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        fine_tuned_results = evaluate_model(
        generator,
        [{"question": sample["question"]} for sample in dataset['train'][:10]]
        )
        filtered_fine_tuned_results = response_val(fine_tuned_results, max_length=150)
        for res in filtered_fine_tuned_results:
            if isinstance(res, dict):
                print(f"Question: {res['question']}\nAnswer: {res['answer']}\nInference Time: {res['inference_time']:.2f} seconds\n")
            else:
                print(f"Answer: {res}\n")
        return generator
    except Exception as e:
        st.error(f"Failed loading fine-tuned model: {e}")
        return None



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
    
    # fine_tune_dataset = convert_qa_to_squad_format('data/QandA.txt', 'data/processed/QA.json')
    print("\n--- Fine-Tuned Model Response Generation Started ---\n")
    print("\n--- Q & A data converstion from .txt to .json format is in process ---\n")
    convert_txt_to_json('data/QandA.txt', 'data/processed/QandA.json')
    print("\n--- Load the Q&A data into input data to model ---\n")
    dataset = load_dataset("json", data_files={"train": "data/processed/QandA.json"})
    # print(f"Fine-tuning dataset created with {len(fine_tune_dataset)} Q/A pairs.")
    test_samples = [
        {
            "question": "What is IRFC’s core business model?",
            "context": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases.",
            "answer": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases."
        },
        {
            "question": "When did IRFC commence project funding to MoR under the finance-lease model?",
            "context": "IRFC commenced project funding to MoR in October 2015; as per a May 23, 2017 MoU with MoR.",
            "answer": "October 2015"
        },
        {
            "question": "How many employees did IRFC have on March 31, 2025?",
            "context": "45 employees; women comprised 20% of the workforce.",
            "answer": "45 employees"
        },
        {
            "question": "What was IRFC’s Revenue from Operations in FY2024-25?",
            "context": "₹27,152.14 crore.",
            "answer": "₹27,152.14 crore"
        },
        {
            "question": "What was Profit After Tax (PAT) in FY2024-25 and growth vs. FY2023-24?",
            "context": "PAT ₹6,502.00 crore, up 1.40% from ₹6,412.11 crore.",
            "answer": "₹6,502.00 crore"
        },
        {
            "question": "What was IRFC’s net worth as on March 31, 2025?",
            "context": "₹52,667.77 crore.",
            "answer": "₹52,667.77 crore"
        },
        {
            "question": "What was the Debt-Equity ratio in FY2024-25?",
            "context": "7.83 times (vs. 8.38 in FY2023-24).",
            "answer": "7.83 times"
        },
        {
            "question": "What were Operating Profit FY2024-25?",
            "context": "Operating Profit 23.93%",
            "answer": "23.93%"
        },
        {
            "question": "Was there any income tax expense in FY2024-25?",
            "context": "zero tax liability due to MAT provisions.",
            "answer": "zero tax liability"
        },
        {
            "question": "What interim dividends did the Board declare in FY2024-25?",
            "context": "Two interim dividends of 8% each (₹0.80 per share) on Nov 4, 2024 (paid Nov 27, 2024) and Mar 17, 2025 (paid Mar 27, 2025).",
            "answer": "Two interim dividends of 8% each (₹0.80 per share)"
        }
    ]

    print("\nBaseline benchmarking with gpt2 pre-trained model:")
    print("-------------------------------------------------------")
    
    # Load Exact Match metric for evaluation
    exact_match_metric = evaluate.load("exact_match")
    f1_metric = evaluate.load("f1")

    # Fine-Tuning
    # * Fine-tune the selected model on your Q/A dataset.
    # * Log all hyperparameters:
    #     * Learning rate, batch size, number of epochs, compute setup (CPU/GPU).
    # * Use efficient techniques as assigned (see next).

    
    # model_name = "gpt2"  # Replace if you want a different model
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.gradient_checkpointing_enable()
    # # GPT2 doesn't have a pad token by default, so we set it
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    # # Select device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # model = model.to(device)
    # # Preprocess the dataset for question answering (SQuAD format)

# Tokenize the dataset, remove original text columns to avoid collator issues
    

    # Compare baseline and fine-tuned results
        # print("\nComparison of Baseline and Fine-Tuned Model Results:")
        # for base, fine in zip(baseline_results, fine_tuned_results):
        #     print(f"Question: {base['question']}")
        #     print(f"Baseline Answer: {base['answer']} | Time: {base['inference_time']:.2f}s")
        #     print(f"Fine-Tuned Answer: {fine['answer']} | Time: {fine['inference_time']:.2f}s")
        #     print("-----") 
        # test_samples = [
        #     {
        #         "question": "What is IRFC’s core business model?",
        #         "context": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases.",
        #         "answer": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases."
        #     },
        #     {
        #         "question": "When did IRFC commence project funding to MoR under the finance-lease model?",
        #         "context": "IRFC commenced project funding to MoR in October 2015; as per a May 23, 2017 MoU with MoR.",
        #         "answer": "October 2015"
        #     },
        #     {
        #         "question": "How many employees did IRFC have on March 31, 2025?",
        #         "context": "45 employees",  
        #         "answer": "45 employees"
        #     }
        # ]
        
        # print("\nBaseline benchmarking with gpt2 pre-trained model:")
        # print("-------------------------------------------------------")
        # qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # baseline_results = evaluate_model(
        #     qa_pipeline,
        #     [{"question": sample["question"]} for sample in test_samples]
        # )
        # for res in baseline_results:
        #     print(f"Question: {res['question']}\nAnswer: {res['answer']}\nInference Time: {res['inference_time']:.2f} seconds\n")
        #     print("-------------------------------------------")
        # print("\nFine-tuned model response generation not implemented yet.")
        # print("-------------------------------------------------------")
        # sample_query = "What was EPS in FY2024-25 vs. FY2023-24?"
        # results = hybrid_retrieve(sample_query, faiss_store, bm25_retriever, top_n=5)
        # print(f"Hybrid Retrieval Results for query '{sample_query}':")
        # for idx, res in enumerate(results):
        #     print(f"Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}")
        # reranked_results = x(sample_query, results, top_k=5)
        # print("\nReranked Results:")
        # for idx, res in enumerate(reranked_results):
        #     print(f"\nReranked Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}\n")
        # filtered_response = response_val(reranked_results, max_length=150)
        # generated_answer = generate_response(llm, filtered_response, reranked_results, max_tokens=1500)
        # print(f"Generated Answer:\n{generated_answer}\n")
        # print("Validating Query and Response...")
        # is_valid, issues = validate_response(generated_answer)
        # if is_valid:
        #     print("Response is valid.")
        # else:
        #     print("Response is invalid. Issues found:")
        #     for issue in issues:
        #         print(f" - {issue}")
        # print("\n--- RAG Pipeline Completed ---\n")

    # Streamlit UI



    #CLI interface using Streamlit
    st.title("RAG System for Financial Document Q&A")
    mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"])
    query = st.text_input("Enter your question")
    if query:
        rag_results = []
        start_time = time.time()
        if mode == "RAG":
            start_rag = time.time()
            print("\n--- RAG Pipeline Started ---\n")
            llm, tokenizer, model, generator = model_creation()
            print("\n--- RAG Model loaded successfully. ---\n")
            # Process the document only once and reuse the vector stores
            # faiss_store, bm25_retriever = process_text(data_preprocess_code)
            top_chunks = hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5)
            print(f"\n--- Top Chunks Retrieved: {len(top_chunks)} ---\n")
            print("\n--- Advanced Reranking with Cross-Encoder... ---\n")
            reranked_results = rerank_with_cross_encoder(query, top_chunks)
            print(f"\n--- Top Chunks after Reranking: {len(reranked_results)} ---\n")
            print("\n--- Generating Response... ---\n")
            rag_answer = generate_response(llm, query, reranked_results)
            rag_confidence = get_confidence_rag(reranked_results)
            print("\n--- Validating Query and Response(Guardrail implementation)... ---\n")
            validation_passed, validation_issues = validate_response(rag_answer)
            rag_time = time.time() - start_rag
            elapsed_time = time.time() - start_time
            rag_correct = correctness(rag_answer, query)
            rag_results.append({
                "Question": query,
                "Method": "RAG",
                "Answer": rag_answer,
                "Confidence": round(rag_confidence, 2),
                "Time (s)": round(rag_time, 2),
                "Correct (Y/N)": rag_correct
            })
            if not validation_passed:
                st.warning("Response validation issues detected:")
                for issue in validation_issues:
                    st.write(f" - {issue}")
            confidence_score = sum([res.metadata.get('similarity', 0) for res in reranked_results]) / len(reranked_results) if reranked_results else 0
            df = pd.DataFrame(rag_results)
            print(df.to_string(index=False))
        else:
            fine_tuned_results = []
            print("\n--- Fine-Tuned Model pre training started ---\n")
            llm, tokenizer, model, generator = model_creation()
            print("\n--- Pre-Trained Fine-Tuned Model loaded successfully. ---\n")
            print("\n--- Training the complete Q&A dataset to Fine-Tuned Model... ---\n")
            QandA_model_train(tokenizer, model, dataset)
            print("\n--- Fine-Tuned Model with Q&A training completed and saved to ./sft_model ---\n")
            start_ft = time.time()
            fine_tuned_generator = load_finetuned_model(dataset)
            print("\n--- Generating Response from Fine-Tuned Model... ---\n")
            ex_response = expert_model_fine_tune(dataset, model, tokenizer, query)
            if fine_tuned_generator:
                answer = generate_response(fine_tuned_generator, query, ex_response, max_tokens=1500)
                st.write(f"Generated Answer:\n{answer}\n")
                confidence_score = 1.0  # Placeholder confidence
            else:
                answer = "Error loading fine-tuned model."
                confidence_score = 0
            ft_time = time.time() - start_ft
            ft_correct = correctness(answer, query)
            fine_tuned_results.append({
                "Question": query,
                "Method": "Fine-Tuned",
                "Answer": answer,
                "Confidence": round(confidence_score, 2),
                "Time (s)": round(ft_time, 2),
                "Correct (Y/N)": ft_correct
            })
            df = pd.DataFrame(fine_tuned_results)
            print(df.to_string(index=False))
            elapsed_time = time.time() - start_time

        if answer:
            st.success(f"Answer: {answer}")
            st.write(f"Confidence Score: {confidence_score:.2f}")
            st.write(f"Method Used: {mode}")
            st.write(f"Response Time: {elapsed_time:.2f} seconds")