# LLM Book Recommender

An intelligent book recommendation system that leverages Large Language Models (LLMs), vector search, and sentiment analysis to provide personalized book recommendations based on user queries, categories, and emotional tones.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
- [Detailed Steps](#detailed-steps)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Kaggle account (for dataset access)
- Kaggle API credentials configured

### Python Libraries
The project uses the following key libraries:

- **Data Processing**: pandas, numpy
- **Machine Learning & LLMs**: 
  - `langchain-huggingface` - For HuggingFace embeddings
  - `langchain-community` - For document loaders
  - `langchain-text-splitters` - For text chunking
  - `langchain-chroma` - For vector database
- **Vector Database**: Chroma - For semantic similarity search
- **UI Framework**: Gradio - For interactive dashboard
- **Data Exploration**: matplotlib, seaborn, jupyter
- **Environment Management**: python-dotenv

### Dataset
- **Source**: [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- **Access**: Downloaded via KaggleHub

## Project Overview

This project implements a multi-step pipeline to build an intelligent book recommendation system:

1. **Text Data Preparation** - Clean and prepare book descriptions for processing
2. **Vector Search** - Convert book descriptions into embeddings and use semantic similarity
3. **Text Classification** - Categorize books using LLM-based classification
4. **Sentiment Analysis** - Extract emotional tones from book descriptions
5. **Interactive Dashboard** - Present recommendations through a user-friendly Gradio interface

## Detailed Steps

### 1. Prepare Text Data
**File**: `data-exploration.ipynb`

This initial step focuses on:
- Loading the Kaggle dataset (7,000+ books with metadata) using `kagglehub`
- Exploring the dataset structure and content (ISBN, title, authors, descriptions, etc.)
- Cleaning and preprocessing book descriptions
- Identifying and handling missing values
- Creating categorized versions of the dataset
- Exporting cleaned data to CSV files for downstream processing

**Output Files**:
- `books_cleaned.csv` - Cleaned book dataset with all preprocessing applied

### 2. Vector Search
**File**: `vector-search.ipynb`

This step implements semantic similarity search:
- Converting book descriptions into dense vector embeddings using HuggingFace embeddings
- Splitting long descriptions into manageable chunks using `CharacterTextSplitter`
- Storing embeddings in Chroma vector database for efficient similarity search
- Creating and testing semantic search queries
- Implementing similarity scoring to rank books by relevance

**Key Concepts**:
- **Embeddings**: Dense vector representations of text that capture semantic meaning
- **Chroma DB**: In-memory vector database for fast similarity searches
- **Similarity Search**: Finding books semantically similar to a user query

**Output Files**:
- Vector database stored in Chroma for fast retrieval

### 3. Text Classification
**File**: `text-classification.ipynb`

This step categorizes books based on their content:
- Using LLMs to extract main topics/genres from book descriptions
- Assigning categories to books (e.g., Fiction, Self-Help, Mystery, etc.)
- Filtering recommendations by selected categories
- Creating category lookup tables for dashboard filtering

**Key Features**:
- Automated topic extraction using language models
- Simplified categories for user-friendly filtering
- Integration with the recommendation pipeline

**Output Files**:
- `books_with_categories.csv` - Books dataset with assigned categories/topics

### 4. Sentiment Analysis
**File**: `sentiment-analysis.ipynb`

This step extracts emotional tones from book descriptions:
- Using LLMs to identify emotional attributes in book descriptions
- Scoring books across emotional dimensions:
  - **Joy**: Positive, uplifting feelings
  - **Surprise**: Unexpected twists and turns
  - **Anger**: Intense, confrontational themes
  - **Fear**: Suspenseful, thrilling elements
  - **Sadness**: Melancholic, emotional depth
- Creating emotion-based filtering for personalized recommendations

**Output Files**:
- `books_with_emotions.csv` - Books dataset with emotional tone scores

### 5. Gradio Dashboard
**File**: `gradio-dashboard.py`

This step provides an interactive user interface:
- Building a semantic book recommendation engine with three filter options:
  - **Query Input**: Natural language description of desired book (e.g., "A story about forgiveness")
  - **Category Filter**: Select from all available book categories
  - **Emotional Tone Filter**: Choose from Happy, Surprising, Angry, Suspenseful, or Sad
- Displaying recommendations with:
  - Book cover thumbnails
  - Title and author information
  - Truncated descriptions
- Implementation details:
  - Retrieves top 50 semantic matches from vector database
  - Filters by category (if selected)
  - Sorts by selected emotional tone
  - Returns top 16 recommendations for gallery display

**How to Run**:
```bash
python gradio-dashboard.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd book-recommender
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install pandas numpy kagglehub
pip install langchain-huggingface langchain-community langchain-text-splitters langchain-chroma
pip install chromadb gradio
pip install matplotlib seaborn jupyter
pip install python-dotenv
```

4. Set up Kaggle API credentials:
- Visit https://www.kaggle.com/account
- Download your API token
- Place it at `~/.kaggle/kaggle.json`

5. Create a `.env` file in the project root for any API keys or configuration

## Usage

1. **Run data exploration** (first time setup):
   ```bash
   jupyter notebook data-exploration.ipynb
   ```

2. **Run vector search setup**:
   ```bash
   jupyter notebook vector-search.ipynb
   ```

3. **Run text classification**:
   ```bash
   jupyter notebook text-classification.ipynb
   ```

4. **Run sentiment analysis**:
   ```bash
   jupyter notebook sentiment-analysis.ipynb
   ```

5. **Launch the dashboard**:
   ```bash
   python gradio-dashboard.py
   ```

The dashboard will be available at `http://localhost:7860` by default.