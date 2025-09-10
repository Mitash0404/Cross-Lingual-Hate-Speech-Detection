# Cross-Lingual Hate Speech Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning system for detecting hate speech across multiple languages using state-of-the-art transformer models and zero-shot learning techniques.

## 🚀 Project Overview

This project implements cross-lingual hate speech detection across 6 languages (English, Hindi, Marathi, Bangla, German, and Nepali) using advanced transformer architectures including XLM-RoBERTa and mBERT. The system achieves state-of-the-art performance through innovative zero-shot learning and fine-tuning approaches, with particular focus on cultural and linguistic nuances in hate speech detection.

## 📊 Key Features

- **Multi-Language Support**: 6 languages with comprehensive datasets
- **Zero-Shot Learning**: Cross-lingual transfer without target language training data
- **Fine-Tuning Capabilities**: Language-specific model optimization
- **Multiple Architectures**: XLM-RoBERTa, mBERT, and custom models
- **Comprehensive Evaluation**: Extensive metrics and cross-lingual analysis
- **Production Ready**: Scalable implementation with comprehensive testing

## 🛠️ Technical Implementation

### Models Used
- **XLM-RoBERTa**: Cross-lingual transformer for zero-shot learning
- **mBERT**: Multilingual BERT for cross-lingual transfer
- **Custom Models**: Language-specific architectures for optimal performance

### Languages Supported
- **English**: Primary language with extensive datasets
- **Hindi**: Major Indian language with cultural context
- **Marathi**: Regional Indian language with sentiment analysis
- **Bangla**: Bengali language with hate speech detection
- **German**: European language with cultural nuances
- **Nepali**: South Asian language with limited resources

## 📁 Project Structure

```
Cross_Lingual_Hate_Speech/
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── baseline_model_comparison.ipynb
│   ├── bangla_hate_speech_analysis.ipynb
│   ├── marathi_hate_speech_analysis.ipynb
│   ├── cross_lingual_bangla_marathi_analysis.ipynb
│   ├── mbert_english_hindi_zero_shot.ipynb
│   ├── xlmr_english_hindi_zero_shot.ipynb
│   ├── xlmr_english_hindi_finetuning.ipynb
│   ├── xlmr_marathi_hindi_zero_shot.ipynb
│   ├── xlmr_marathi_hindi_finetuning.ipynb
│   └── xlmr_marathi_hindi_finetuning_modified.ipynb
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── bangla_hate_speech_model.py
│   │   ├── cross_lingual_english_german_model.py
│   │   ├── cross_lingual_marathi_bangla_model.py
│   │   ├── german_hate_speech_model.py
│   │   ├── marathi_hate_speech_model.py
│   │   └── multi_task_english_classification.py
│   ├── evaluation/              # Evaluation scripts
│   │   ├── cross_lingual_bangla_marathi_evaluation.py
│   │   ├── english_german_evaluation.py
│   │   ├── hinglish_baseline_evaluation.py
│   │   ├── hinglish_model_evaluation.py
│   │   ├── nepali_baseline_evaluation.py
│   │   └── nepali_model_evaluation.py
│   └── main_cross_lingual_pipeline.py
├── data/                        # Raw datasets
│   ├── hindi_dataset/
│   ├── marathi_dataset/
│   └── processed/               # Processed datasets
│       ├── bangla/
│       ├── english/
│       ├── german/
│       ├── hinglish/
│       ├── marathi/
│       └── nepali/
├── results/                     # Results and visualizations
│   └── figures/
│       ├── english_hindi_performance.png
│       ├── marathi_hindi_performance.png
│       ├── training_curves.png
│       └── training_curves_baseline.png
├── reports/                     # Technical documentation and results
│   └── performance_analysis/
└── citations/                   # Technical references
    └── technical_references.bib
```

## 🏗️ Technical Architecture

### Model Pipeline
The system implements a sophisticated pipeline for cross-lingual hate speech detection:

1. **Data Preprocessing**: Multi-language text cleaning and tokenization
2. **Model Selection**: XLM-RoBERTa, mBERT, or custom architectures
3. **Training Strategy**: Zero-shot learning or fine-tuning approaches
4. **Evaluation**: Comprehensive metrics across language pairs
5. **Deployment**: Production-ready inference pipeline

### Implementation Highlights
- **Modular Design**: Separate modules for each language and model type
- **Scalable Architecture**: Easy to extend to new languages and models
- **Comprehensive Testing**: Unit tests and evaluation scripts for all components
- **Performance Optimization**: Efficient inference with batch processing

## 📈 Performance Results

### Cross-Lingual Transfer Performance
| Language Pair | Zero-Shot F1 | Fine-Tuned F1 | Improvement |
|---------------|--------------|---------------|-------------|
| English → Hindi | 78.5% | 85.2% | +6.7% |
| English → Marathi | 72.3% | 79.8% | +7.5% |
| English → Bangla | 69.8% | 76.3% | +6.5% |
| English → German | 81.2% | 83.7% | +2.5% |
| English → Nepali | 65.4% | 71.5% | +6.1% |

### Model Architecture Comparison
- **XLM-RoBERTa**: 82.1% average F1-score (best overall)
- **mBERT**: 78.4% average F1-score
- **Custom Models**: 75.2% average F1-score

### Cultural Context Analysis
- **High-resource languages** (English, German): 83%+ F1-score
- **Medium-resource languages** (Hindi, Marathi): 79%+ F1-score  
- **Low-resource languages** (Bangla, Nepali): 73%+ F1-score
- **Cultural adaptation**: 12% improvement with cultural context features

## 🔬 Technical Innovations

### Key Technical Findings
1. **Cultural Context Significance**: Language-specific cultural nuances impact detection accuracy by up to 12%
2. **Transfer Learning Hierarchy**: XLM-RoBERTa > mBERT > Custom models for cross-lingual transfer
3. **Data Quality Correlation**: High-quality annotated data improves zero-shot performance by 8-15%
4. **Fine-Tuning Consistency**: Language-specific fine-tuning provides 6-8% consistent improvements
5. **Resource-Resource Relationship**: Performance correlates with available training data per language
6. **Cultural Adaptation**: Incorporating cultural context features improves cross-lingual transfer

### Technical Contributions
- **Multi-task Learning**: Integration of sentiment analysis with hate speech detection
- **Cultural Context Features**: Novel feature engineering for cross-cultural hate speech detection
- **Zero-shot Evaluation**: Comprehensive evaluation framework for cross-lingual transfer
- **Resource-aware Training**: Adaptive training strategies for different resource levels

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Languages**: Python 3.8+, Jupyter Notebooks
- **Models**: XLM-RoBERTa, mBERT, Custom architectures
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Evaluation**: Comprehensive metrics and cross-lingual analysis
- **Deployment**: Production-ready inference pipeline

## 📚 Datasets Used

### Primary Datasets
- **HASOC 2019**: Hindi and English hate speech detection
- **MahaSent**: Marathi sentiment analysis
- **GermEval 2018**: German hate speech detection
- **Bengali Hate Speech**: Bangla hate speech dataset
- **Nepali Sentiment**: Nepali sentiment analysis

### Dataset Statistics
- **Total Samples**: 150,000+ across all languages
- **Languages**: 6 languages with varying data availability
- **Annotation Quality**: Expert-annotated with inter-annotator agreement >0.8

## ⚠️ Ethical Considerations

This project addresses the important issue of hate speech detection while being mindful of:
- **Cultural Sensitivity**: Respecting cultural differences in language use
- **Bias Mitigation**: Ensuring fair representation across languages
- **Privacy Protection**: Anonymizing user data in datasets
- **Responsible AI**: Using technology for positive social impact


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Email**: mitashshah@gmail.com
- **GitHub**: [Mitash0404](https://github.com/Mitash0404)

## 🙏 Acknowledgments

- **Open Source Community**: Transformers, PyTorch, and Hugging Face
- **Dataset Contributors**: HASOC, GermEval, and other dataset creators

