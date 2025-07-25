"""
models.py - AI Models Module

Chứa các model AI cho hệ thống trắc nghiệm:
1. DifficultyClassifier: Phân loại độ khó câu hỏi (easy/medium/hard)
   - Sử dụng Random Forest + TF-IDF
   - 15 features engineering từ NLP analysis
   - Train/test split 80/20 với evaluation metrics

2. SimilarQuestionFinder: Tìm câu hỏi tương tự
   - TF-IDF vectorization + Cosine similarity
   - Train trên subset, evaluate subject accuracy
   - Cache vectors cho performance

Tối ưu tốc độ với Streamlit caching và reduced complexity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
from vietnamese_nlp import clean_vietnamese_text

class DifficultyClassifier:
    def __init__(self):
        # Reduced complexity for speed
        self.text_vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        # Use only Random Forest for speed
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def _extract_advanced_features(self, question_text, options_text):
        """Trích xuất đặc trưng nâng cao - tối ưu tốc độ"""
        full_text = question_text + " " + options_text
        full_lower = full_text.lower()
        
        # Pre-compiled patterns for speed
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
        
        # Fast keyword counting
        for category in ['analysis', 'calculation', 'synthesis', 'evaluation', 'definition', 'identification']:
            count = sum(1 for word in self._patterns[category] if word in full_lower)
            features.append(count)
        
        # Basic linguistic features
        features.extend([
            len(question_text.split()),
            len(options_text.split()),
            full_text.count('.'),
            sum(1 for c in full_text if c in '+-*/=()$^_'),
            sum(1 for c in full_text if c.isupper()),
            sum(1 for w in full_text.split() if len(w) > 8)
        ])
        
        # Question type (simplified)
        features.extend([
            1 if 'tại sao' in full_lower or 'vì sao' in full_lower else 0,
            1 if 'như thế nào' in full_lower else 0,
            1 if 'bao nhiêu' in full_lower else 0
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _create_sophisticated_labels(self, data):
        """Tạo nhãn - tối ưu tốc độ"""
        difficulties = []
        
        for _, row in data.iterrows():
            options_text = ' '.join(row['options']) if row['options'] else ''
            features = self._extract_advanced_features(row['question'], options_text).flatten()
            
            # Simplified scoring
            score = (features[0] + features[1] + features[2]) * 2 - features[4] - features[5]
            score += features[6] * 0.1 + features[9] * 0.2 + features[12] + features[13] * 2
            
            if score <= 2:
                difficulty = 'easy'
            elif score <= 5:
                difficulty = 'medium'  
            else:
                difficulty = 'hard'
            
            difficulties.append(difficulty)
        
        return difficulties
    
    def train(self, data):
        """Huấn luyện model với train/test split"""
        # Create sophisticated labels
        difficulties = self._create_sophisticated_labels(data)
        
        # Prepare text features
        texts = []
        for _, row in data.iterrows():
            options_text = ' '.join(row['options']) if row['options'] else ''
            full_text = row['question'] + ' ' + options_text
            processed_text = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            texts.append(processed_text)
        
        # Train/Test Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, difficulties, 
            test_size=0.2, 
            random_state=42, 
            stratify=difficulties
        )
        
        # Vectorize
        X_train_vec = self.text_vectorizer.fit_transform(X_train_text)
        X_test_vec = self.text_vectorizer.transform(X_test_text)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Difficulty Classification Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Easy - Precision: {report['easy']['precision']:.3f}, Recall: {report['easy']['recall']:.3f}")
        print(f"Medium - Precision: {report['medium']['precision']:.3f}, Recall: {report['medium']['recall']:.3f}")
        print(f"Hard - Precision: {report['hard']['precision']:.3f}, Recall: {report['hard']['recall']:.3f}")
        
        self.is_trained = True
        return difficulties

class SimilarQuestionFinder:
    def __init__(self, data):
        self.data = data
        # Split data for training similarity model
        self.train_data, self.test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data['subject']
        )
        
        # Simplified TF-IDF for speed
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.question_vectors = None
        self._prepare_vectors()
        self._evaluate_similarity()
    
    def _prepare_vectors(self):
        """Chuẩn bị vectors - train trên train_data"""
        processed_questions = []
        
        # Use only training data for fitting
        for _, row in self.train_data.iterrows():
            full_text = row['question'] + ' ' + ' '.join(row['options']) if row['options'] else row['question']
            processed = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            processed_questions.append(processed)
        
        # Fit vectorizer on training data
        self.vectorizer.fit(processed_questions)
        
        # Transform all data (for similarity search)
        all_processed = []
        for _, row in self.data.iterrows():
            full_text = row['question'] + ' ' + ' '.join(row['options']) if row['options'] else row['question']
            processed = clean_vietnamese_text(full_text, remove_stopwords=True, normalize=True)
            all_processed.append(processed)
            
        self.question_vectors = self.vectorizer.transform(all_processed)
    
    def _evaluate_similarity(self):
        """Đánh giá chất lượng similarity trên test set"""
        correct_same_subject = 0
        total_tests = 0
        
        # Test trên một số câu hỏi từ test set
        test_sample = self.test_data.sample(min(50, len(self.test_data)), random_state=42)
        
        for _, test_question in test_sample.iterrows():
            similar_questions = self.find_similar_questions(
                test_question['id'], n_similar=3, same_subject_only=False
            )
            
            if similar_questions:
                # Kiểm tra xem câu hỏi tương tự nhất có cùng subject không
                most_similar = similar_questions[0]
                if most_similar['question_data']['subject'] == test_question['subject']:
                    correct_same_subject += 1
                total_tests += 1
        
        if total_tests > 0:
            subject_accuracy = correct_same_subject / total_tests
            print(f"Similar Question Finder Results:")
            print(f"Subject Accuracy: {subject_accuracy:.3f}")
            print(f"Test samples: {total_tests}")
        
    def find_similar_questions(self, current_question_id, n_similar=3, same_subject_only=True):
        """Tìm câu hỏi tương tự - tối ưu tốc độ"""
        try:
            current_idx = None
            current_subject = None
            
            # Fast lookup using pandas query
            for idx, (_, row) in enumerate(self.data.iterrows()):
                if row['id'] == current_question_id:
                    current_idx = idx
                    current_subject = row['subject']
                    break
            
            if current_idx is None:
                return []
            
            # Simple cosine similarity
            current_vector = self.question_vectors[current_idx]
            similarities = cosine_similarity(current_vector, self.question_vectors).flatten()
            
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
            
            # Simple sorting by similarity only
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_questions[:n_similar]
            
        except Exception as e:
            return []

@st.cache_resource
def initialize_models(data):
    """Initialize models with caching"""
    # Make a copy to avoid modifying cached data
    data_copy = data.copy()
    
    # Train difficulty classifier
    difficulty_classifier = DifficultyClassifier()
    difficulties = difficulty_classifier.train(data_copy)
    
    # Add difficulty column to data copy
    data_copy['difficulty'] = difficulties
    
    # Initialize similar question finder with updated data
    similar_finder = SimilarQuestionFinder(data_copy)
    
    return difficulty_classifier, difficulties, similar_finder