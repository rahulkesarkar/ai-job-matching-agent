# AI Job-Matching Agent

An end-to-end AI/ML application that matches candidate
profiles with relevant job opportunities using NLP,
semantic search, vector retrieval, and hybrid ranking.

## Project Objective

Build an intelligent job-matching system that can:

- Parse candidate resumes.
- Extract structured candidate profiles.
- Understand job descriptions using NLP.
- Generate semantic embeddings.
- Retrieve relevant jobs using vector similarity.
- Rank opportunities using a hybrid ranking system.
- Generate evidence-based match explanations.
- Identify candidate skill gaps.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Sentence Transformers
- FAISS
- Large Language Model API
- FastAPI
- Streamlit
- Docker
- AWS
- Git

## Project Status

Development in progress.

## Architecture

The application will use a modular architecture
separating data processing, ML services, API endpoints,
and the user interface.

Detailed architecture and design decisions will be
documented as development progresses.

## Evaluation

The retrieval and ranking system will be evaluated
using appropriate information retrieval metrics,
including Precision@K, Recall@K, MRR, and NDCG.

## Setup

Installation and execution instructions will be added
as the application components are implemented.

## License

To be determined.
EOFC
cat > README.md <<'EOF'
# AI Job-Matching Agent

I'm building this project to explore how machine learning and NLP can make job searching more relevant.

Most basic job portals rely heavily on keywords. This can be limiting because a candidate might have the right experience and skills but describe them differently from the job description.

The idea behind this project is to build a system that looks beyond exact keyword matches and understands how a candidate's experience relates to a job's requirements.

## What I'm trying to build

A user should be able to upload their resume and get a ranked list of jobs that are relevant to their profile.

The system will:

- Extract relevant information from a resume, including skills, education and experience.
- Process job descriptions and identify their requirements.
- Use text embeddings to understand the similarity between candidate profiles and job descriptions.
- Retrieve relevant opportunities using vector search.
- Rank jobs using a combination of semantic similarity and skill matching.
- Generate explanations for each match and highlight skills the candidate may be missing.

## My approach

I'll start with a traditional text-matching approach using TF-IDF as a baseline.

After that, I'll implement semantic matching using Sentence Transformers and FAISS. I'll compare both approaches to understand whether semantic search actually improves job retrieval.

The ranking system will combine multiple signals rather than relying entirely on an LLM-generated score. I'll also evaluate the results using metrics such as Precision@K, Recall@K, MRR and NDCG.

## Technologies

- Python
- Pandas
- Scikit-learn
- Sentence Transformers
- FAISS
- Claude API or similar LLM API
- FastAPI
- Streamlit
- Docker
- AWS
- Git and GitHub

## Project Structure

The application is being developed as a modular project, with separate components for data processing, resume parsing, matching, ranking and API development.

## Current Status

The project is currently under development. I've set up the Python environment, created the initial project structure and initialized Git.

The next step is to build the job dataset and start working on the matching engine.

I'll update this README as I implement and test each component.
