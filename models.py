"""
models.py - AI Models Module with Enhanced Evaluation

LOGIC CHÍNH:
1. DifficultyClassifier: Phân loại độ khó câu hỏi (easy/medium/hard)
   - Random Forest + TF-IDF với 15 features NLP
   - Enhanced evaluation: F1-score, Confusion Matrix, Cross-validation
   - Train/test split với detailed metrics

2. SimilarQuestionFinder: Tìm câu hỏi tương tự
   - TF-IDF vectorization + Cosine similarity
   - Enhanced evaluation: Subject accuracy, Cross-subject testing

EVALUATION ENHANCEMENTS:
- F1-score (macro, micro, weighted) cho từng class
- Confusion matrix với visualization-ready format
- 5-fold cross-validation với mean±std metrics
- Classification report chi tiết
- Model performance summary
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score
)
import streamlit as st
from vietnamese_nlp import clean_vietnamese_text

class DifficultyClassifier:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.evaluation_results = {}
    
    def _extract_features(self, question_text, options_text):
        """Trích xuất 15 features từ văn bản"""
        full_text = question_text + " " + options_text
        full_lower = full_text.lower()
        
        # Pre-compiled keyword patterns
        if not hasattr(self, '_patterns'):
            self._patterns = {
                'analysis': ['phân tích', 'so sánh', 'đánh giá', 'giải thích'],
                'calculation': ['tính', 'toán', 'công thức', 'mol', 'gam'],
                'synthesis': ['tổng hợp', 'phản ứng', 'cơ chế', 'quá trình'],
                'evaluation': ['ảnh hưởng', 'tác động', 'nguyên nhân'],
                'definition': ['là gì', 'tên gọi', 'thuộc'],
                'identification': ['màu', 'trạng thái', 'tính chất']
            }
        
        features = []
        
        # Keyword counting (6 features)
        for category in ['analysis', 'calculation', 'synthesis', 'evaluation', 'definition', 'identification']:
            count = sum(1 for word in self._patterns[category] if word in full_lower)
            features.append(count)
        
        # Linguistic features (6 features)
        features.extend([
            len(question_text.split()),  # question length
            len(options_text.split()),   # options length
            full_text.count('.'),        # sentence count
            sum(1 for c in full_text if c in '+-*/=()$^_'),  # math symbols
            sum(1 for c in full_text if c.isupper()),        # uppercase count
            sum(1 for w in full_text.split() if len(w) > 8)  # long words
        ])
        
        # Question type features (3 features)
        features.extend([
            1 if 'tại sao' in full_lower or 'vì sao' in full_lower else 0,
            1 if 'như thế nào' in full_lower else 0,
            1 if 'bao nhiêu' in full_lower else 0
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _create_labels(self, data):
        """Tạo nhãn độ khó dựa trên rule-based scoring"""
        difficulties = []
        
        for _, row in data.iterrows():
            options_text = ' '.join(row['options']) if row['options'] else ''
            features = self._extract_features(row['question'], options_text).flatten()
            
            # Weighted scoring
            score = (features[0] + features[1] + features[2]) * 2  # analysis keywords
            score -= (features[4] + features[5])                  # definition keywords
            score += features[6] * 0.1 + features[9] * 0.2        # text complexity
            score += features[12] + features[13] * 2               # question complexity
            
            if score <= 2:
                difficulty = 'easy'
            elif score <= 5:
                difficulty = 'medium'  
            else:
                difficulty = 'hard'
            
            difficulties.append(difficulty)
        
        return difficulties
    
    def _evaluate_model_comprehensive(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation với F1-score, Confusion Matrix, Cross-validation"""
        
        # 1. Basic predictions
        y_pred = self.model.predict(X_test)
        
        # 2. Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # 3. F1-scores (macro, micro, weighted)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 4. Precision and Recall (macro averages)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        # 5. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y_test) | set(y_pred)))
        
        # 6. Classification Report (detailed per-class metrics)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # 7. Cross-Validation (5-fold)
        cv_scores = self._perform_cross_validation(X_train, y_train)
        
        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'confusion_matrix_labels': labels,
            'classification_report': class_report,
            'cross_validation': cv_scores
        }
        
        return self.evaluation_results
    
    def _perform_cross_validation(self, X, y, cv_folds=5):
        """5-fold cross-validation"""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        cv_f1_macro = cross_val_score(self.model, X, y, cv=skf, scoring='f1_macro')
        cv_f1_weighted = cross_val_score(self.model, X, y, cv=skf, scoring='f1_weighted')
        cv_precision = cross_val_score(self.model, X, y, cv=skf, scoring='precision_macro')
        cv_recall = cross_val_score(self.model, X, y, cv=skf, scoring='recall_macro')
        
        cv_results = {
            'accuracy': {
                'scores': cv_accuracy,
                'mean': cv_accuracy.mean(),
                'std': cv_accuracy.std()
            },
            'f1_macro': {
                'scores': cv_f1_macro,
                'mean': cv_f1_macro.mean(),
                'std': cv_f1_macro.std()
            },
            'f1_weighted': {
                'scores': cv_f1_weighted,
                'mean': cv_f1_weighted.mean(),
                'std': cv_f1_weighted.std()
            },
            'precision_macro': {
                'scores': cv_precision,
                'mean': cv_precision.mean(),
                'std': cv_precision.std()
            },
            'recall_macro': {
                'scores': cv_recall,
                'mean': cv_recall.mean(),
                'std': cv_recall.std()
            }
        }
        
        return cv_results
    
    def _print_evaluation_results(self):
        """In kết quả evaluation chi tiết"""
        results = self.evaluation_results
        
        print("=" * 60)
        print("DIFFICULTY CLASSIFICATION - COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Basic metrics
        print(f"\n📊 BASIC METRICS:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro): {results['recall_macro']:.4f}")
        
        # F1-scores
        print(f"\n🎯 F1-SCORES:")
        print(f"F1-Score (macro): {results['f1_macro']:.4f}")
        print(f"F1-Score (micro): {results['f1_micro']:.4f}")
        print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
        
        # Per-class metrics
        print(f"\n📋 PER-CLASS METRICS:")
        class_report = results['classification_report']
        for label in ['easy', 'medium', 'hard']:
            if label in class_report:
                print(f"{label.upper():<8} - Precision: {class_report[label]['precision']:.4f}, "
                      f"Recall: {class_report[label]['recall']:.4f}, "
                      f"F1-Score: {class_report[label]['f1-score']:.4f}, "
                      f"Support: {class_report[label]['support']}")
        
        # Confusion Matrix
        print(f"\n🔢 CONFUSION MATRIX:")
        cm = results['confusion_matrix']
        labels = results['confusion_matrix_labels']
        
        print(f"{'':>8}", end="")
        for label in labels:
            print(f"{label:>8}", end="")
        print()
        
        for i, true_label in enumerate(labels):
            print(f"{true_label:>8}", end="")
            for j in range(len(labels)):
                print(f"{cm[i][j]:>8}", end="")
            print()
        
        # Cross-validation results
        print(f"\n🔄 CROSS-VALIDATION (5-fold):")
        cv = results['cross_validation']
        print(f"Accuracy: {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}")
        print(f"F1-Score (macro): {cv['f1_macro']['mean']:.4f} ± {cv['f1_macro']['std']:.4f}")
        print(f"F1-Score (weighted): {cv['f1_weighted']['mean']:.4f} ± {cv['f1_weighted']['std']:.4f}")
        print(f"Precision (macro): {cv['precision_macro']['mean']:.4f} ± {cv['precision_macro']['std']:.4f}")
        print(f"Recall (macro): {cv['recall_macro']['mean']:.4f} ± {cv['recall_macro']['std']:.4f}")
        
        print("=" * 60)
    
    def train(self, data):
        """Huấn luyện model với comprehensive evaluation"""
        difficulties = self._create_labels(data)
        
        # Prepare text features
        texts = []
        for _, row in data.iterrows():
            options_text = ' '.join(row['options']) if row['options'] else ''
            full_text = row['question'] + ' ' + options_text
            processed_text = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            texts.append(processed_text)
        
        # Train/Test Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, difficulties, test_size=0.2, random_state=42, stratify=difficulties
        )
        
        # Vectorize
        X_train_vec = self.text_vectorizer.fit_transform(X_train_text)
        X_test_vec = self.text_vectorizer.transform(X_test_text)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Comprehensive evaluation
        self._evaluate_model_comprehensive(X_train_vec, X_test_vec, y_train, y_test)
        
        # Print results
        self._print_evaluation_results()
        
        self.is_trained = True
        return difficulties

class SimilarQuestionFinder:
    def __init__(self, data):
        self.data = data
        
        # Train/test split for evaluation
        self.train_data, self.test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data['subject']
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        
        self.question_vectors = None
        self.evaluation_results = {}
        self._prepare_vectors()
        self._evaluate_similarity_comprehensive()
    
    def _prepare_vectors(self):
        """Chuẩn bị vectors - fit trên train data"""
        # Fit vectorizer trên training data
        train_texts = []
        for _, row in self.train_data.iterrows():
            full_text = row['question'] + ' ' + ' '.join(row['options']) if row['options'] else row['question']
            processed = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            train_texts.append(processed)
        
        self.vectorizer.fit(train_texts)
        
        # Transform tất cả data để tìm similarity
        all_texts = []
        for _, row in self.data.iterrows():
            full_text = row['question'] + ' ' + ' '.join(row['options']) if row['options'] else row['question']
            processed = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            all_texts.append(processed)
            
        self.question_vectors = self.vectorizer.transform(all_texts)
    
    def _evaluate_similarity_comprehensive(self):
        """Enhanced evaluation với multiple metrics"""
        
        # Test trên test set
        test_sample = self.test_data.sample(min(100, len(self.test_data)), random_state=42)
        
        # Metrics tracking
        same_subject_correct = 0
        cross_subject_similarity = []
        within_subject_similarity = []
        total_tests = 0
        
        # Subject-wise accuracy
        subject_accuracy = {}
        subjects = test_sample['subject'].unique()
        for subject in subjects:
            subject_accuracy[subject] = {'correct': 0, 'total': 0}
        
        for _, test_question in test_sample.iterrows():
            similar_questions = self.find_similar_questions(
                test_question['id'], n_similar=5, same_subject_only=False
            )
            
            if similar_questions:
                # Overall accuracy
                most_similar = similar_questions[0]
                if most_similar['question_data']['subject'] == test_question['subject']:
                    same_subject_correct += 1
                
                # Subject-specific accuracy
                subject = test_question['subject']
                subject_accuracy[subject]['total'] += 1
                if most_similar['question_data']['subject'] == subject:
                    subject_accuracy[subject]['correct'] += 1
                
                # Similarity score analysis
                for similar in similar_questions:
                    sim_score = similar['similarity']
                    if similar['question_data']['subject'] == test_question['subject']:
                        within_subject_similarity.append(sim_score)
                    else:
                        cross_subject_similarity.append(sim_score)
                
                total_tests += 1
        
        # Calculate final metrics
        overall_subject_accuracy = same_subject_correct / total_tests if total_tests > 0 else 0
        
        # Calculate subject-specific accuracies
        for subject in subject_accuracy:
            if subject_accuracy[subject]['total'] > 0:
                subject_accuracy[subject]['accuracy'] = (
                    subject_accuracy[subject]['correct'] / subject_accuracy[subject]['total']
                )
            else:
                subject_accuracy[subject]['accuracy'] = 0
        
        # Store results
        self.evaluation_results = {
            'overall_subject_accuracy': overall_subject_accuracy,
            'subject_specific_accuracy': subject_accuracy,
            'within_subject_similarity': {
                'mean': np.mean(within_subject_similarity) if within_subject_similarity else 0,
                'std': np.std(within_subject_similarity) if within_subject_similarity else 0,
                'count': len(within_subject_similarity)
            },
            'cross_subject_similarity': {
                'mean': np.mean(cross_subject_similarity) if cross_subject_similarity else 0,
                'std': np.std(cross_subject_similarity) if cross_subject_similarity else 0,
                'count': len(cross_subject_similarity)
            },
            'total_tests': total_tests
        }
        
        self._print_similarity_evaluation()
    
    def _print_similarity_evaluation(self):
        """In kết quả evaluation chi tiết cho similarity finder"""
        results = self.evaluation_results
        
        print("=" * 60)
        print("SIMILAR QUESTION FINDER - COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"Subject Accuracy: {results['overall_subject_accuracy']:.4f}")
        print(f"Total Test Samples: {results['total_tests']}")
        
        print(f"\n📋 SUBJECT-SPECIFIC ACCURACY:")
        for subject, metrics in results['subject_specific_accuracy'].items():
            print(f"{subject.upper():<12} - Accuracy: {metrics['accuracy']:.4f} "
                  f"({metrics['correct']}/{metrics['total']})")
        
        print(f"\n📈 SIMILARITY SCORE ANALYSIS:")
        within = results['within_subject_similarity']
        cross = results['cross_subject_similarity']
        
        print(f"Within Subject - Mean: {within['mean']:.4f} ± {within['std']:.4f} "
              f"(n={within['count']})")
        print(f"Cross Subject  - Mean: {cross['mean']:.4f} ± {cross['std']:.4f} "
              f"(n={cross['count']})")
        
        # Interpretation
        if within['mean'] > cross['mean']:
            print(f"✅ Good separation: Within-subject similarity > Cross-subject similarity")
        else:
            print(f"⚠️  Poor separation: Need to improve subject discrimination")
        
        print("=" * 60)
    
    def find_similar_questions(self, current_question_id, n_similar=3, same_subject_only=True):
        """Tìm câu hỏi tương tự"""
        try:
            # Tìm index của câu hỏi hiện tại
            current_idx = None
            current_subject = None
            
            for idx, (_, row) in enumerate(self.data.iterrows()):
                if row['id'] == current_question_id:
                    current_idx = idx
                    current_subject = row['subject']
                    break
            
            if current_idx is None:
                return []
            
            # Tính cosine similarity
            current_vector = self.question_vectors[current_idx]
            similarities = cosine_similarity(current_vector, self.question_vectors).flatten()
            
            # Collect similar questions
            similar_questions = []
            for idx, similarity in enumerate(similarities):
                if idx != current_idx:
                    question_data = self.data.iloc[idx]
                    
                    if same_subject_only and question_data['subject'] != current_subject:
                        continue
                    
                    similar_questions.append({
                        'question_data': question_data,
                        'similarity': similarity,
                        'index': idx
                    })
            
            # Sort by similarity
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_questions[:n_similar]
            
        except Exception:
            return []

@st.cache_resource
def initialize_models(data):
    """Initialize models với caching"""
    data_copy = data.copy()
    
    # Train difficulty classifier với enhanced evaluation
    difficulty_classifier = DifficultyClassifier()
    difficulties = difficulty_classifier.train(data_copy)
    
    # Add difficulty column
    data_copy['difficulty'] = difficulties
    
    # Initialize similar question finder với enhanced evaluation
    similar_finder = SimilarQuestionFinder(data_copy)
    
    return difficulty_classifier, difficulties, similar_finder