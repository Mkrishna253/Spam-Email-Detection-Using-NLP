# 📧 Spam Email Detection Using NLP

This repository presents a comprehensive machine learning pipeline for classifying emails as spam or not spam (ham) using natural language processing (NLP) techniques. The project demonstrates best practices in data preprocessing, feature engineering, model training, and evaluation, with a focus on both classical and modern NLP approaches.

## ✨ Features
- Complete workflow from raw data to model evaluation
- Text preprocessing: lowercasing, punctuation and whitespace removal
- Feature extraction using TF-IDF (unigrams and bigrams)
- Multiple machine learning models: Naive Bayes, Logistic Regression, Support Vector Machine
- Visualizations: class distribution, confusion matrices
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- BERT transformer model integration (work in progress)

## 📝 Problem Statement
Unsolicited spam emails are a persistent problem, posing security risks and reducing productivity. This project aims to automate the detection of spam emails, enabling users and organizations to filter unwanted messages and mitigate risks efficiently.

## 📂 Dataset
- Email text content
- Label: Spam (1) or Ham (0)
- Source: `combined_data.csv`

## 🔧 Project Workflow
1. **Exploratory Data Analysis (EDA):** Assess class balance and visualize data distributions
2. **Preprocessing:** Clean and normalize text data
3. **Feature Engineering:** Convert text to numerical features using TF-IDF
4. **Model Training:** Train and compare Naive Bayes, Logistic Regression, and SVM classifiers
5. **Evaluation:** Generate classification reports and confusion matrices for model comparison
6. **Advanced NLP:** Integrate BERT transformer model for state-of-the-art performance (in progress)

## 🚀 Getting Started
1. Clone this repository
2. Install dependencies from `requirements.txt`
3. Ensure `combined_data.csv` is present in the project directory
4. Run the notebook `spam_classifier.ipynb` to reproduce results and visualizations

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Transformers (HuggingFace)

## 📊 Results
The project benchmarks multiple models and provides detailed evaluation metrics. Visualizations help interpret model performance and dataset characteristics. The integration of BERT aims to further improve classification accuracy using advanced NLP techniques.

## 📄 License
This project is provided for educational and research purposes. Please refer to the repository license for usage details.

## 🙏 Acknowledgements
- Scikit-learn documentation
- HuggingFace Transformers
- Open-source datasets and contributors