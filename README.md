#  Cross-Domain Fake Review Detection (CLI Version)

##  Objective
Detect **fake vs genuine reviews** across **multiple domains** (Amazon, Hotels, Yelp)  
and evaluate using **Leave-One-Domain-Out** strategy.

##  Datasets
- Amazon Fine Food Reviews (~500k reviews)
- Deceptive Opinion Spam Corpus (truthful vs deceptive hotel reviews)
- Yelp Labelled Dataset (spam vs genuine reviews)

##  Features
- Preprocessing: lowercasing, stopword removal, tokenization  
- Feature extraction: TF-IDF  
- Model: Logistic Regression  
- Evaluation: Accuracy, Precision, Recall, F1  
- Interface: Command-Line (CLI)

##  Run Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run main program
python main.py
