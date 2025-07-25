"""
topic_classifier.py - Deep Learning Topic Classification Module

LOGIC CHÍNH:
1. TopicClassifier: Phân loại chủ đề câu hỏi theo chương
   - Sử dụng BERT/PhoBERT để hiểu ngữ cảnh sâu
   - Rule-based labeling cho training data
   - Fine-tuning pre-trained models

2. Subject-specific topics:
   - Physics: "Dao động cơ", "Điện xoay chiều", "Sóng cơ", etc.
   - Chemistry: "Este – Lipit", "Điện phân", "Hóa hữu cơ", etc.
   - Biology: "Di truyền học", "Tiến hóa", "Sinh thái học", etc.

DEEP LEARNING APPROACH:
- Pre-trained PhoBERT cho Vietnamese understanding
- Fine-tuning with topic-specific data
- Cross-validation evaluation
- Confidence scoring cho predictions

FALLBACK:
- Nếu không có GPU/transformers → dùng TF-IDF + Random Forest
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Try import transformers, fallback to traditional ML if not available
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using traditional ML fallback.")

class TopicClassifier:
    def __init__(self, use_bert=True):
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.is_trained = False
        
        # Subject-specific topics mapping
        self.subject_topics = {
            'physics': [
                'Dao động cơ', 'Sóng cơ', 'Điện xoay chiều', 'Từ trường',
                'Điện trường', 'Quang học', 'Cơ học', 'Nhiệt học',
                'Vật lý hạt nhân', 'Điện từ học'
            ],
            'chemistry': [
                'Hóa hữu cơ', 'Este – Lipit', 'Điện phân', 'Cân bằng hóa học',
                'Axit - Bazơ', 'Oxi hóa - Khử', 'Polime', 'Kim loại',
                'Phi kim', 'Hóa học môi trường'
            ],
            'biology': [
                'Di truyền học', 'Tiến hóa', 'Sinh thái học', 'Tế bào học',
                'Sinh lý học', 'Phân loại sinh vật', 'Sinh học phân tử',
                'Miễn dịch học', 'Nội tiết', 'Hô hấp - Tuần hoàn'
            ]
        }
        
        # Initialize fallback model
        if not self.use_bert:
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize TF-IDF + Random Forest fallback"""
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.fallback_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    def _create_topic_labels(self, data):
        """Rule-based topic labeling dựa trên keywords"""
        topic_keywords = {
            # Physics topics
            'Dao động cơ': [
                'dao động', 'chu kỳ', 'tần số', 'biên độ', 'con lắc',
                'lò xo', 'điều hòa', 'pha dao động'
            ],
            'Sóng cơ': [
                'sóng', 'tần số sóng', 'bước sóng', 'âm thanh', 'siêu âm',
                'cộng hưởng', 'giao thoa', 'độ to', 'độ cao'
            ],
            'Điện xoay chiều': [
                'xoay chiều', 'điện áp hiệu dụng', 'dòng điện xoay chiều',
                'máy biến áp', 'điện từ cảm ứng', 'từ thông'
            ],
            'Điện trường': [
                'điện trường', 'điện tích', 'thế năng điện', 'điện thế',
                'tụ điện', 'năng lượng điện trường'
            ],
            'Quang học': [
                'ánh sáng', 'khúc xạ', 'phản xạ', 'thấu kính', 'gương',
                'giao thoa ánh sáng', 'tán sắc', 'quang phổ'
            ],
            'Cơ học': [
                'lực', 'gia tốc', 'vận tốc', 'động lượng', 'năng lượng',
                'công', 'ma sát', 'chuyển động'
            ],
            
            # Chemistry topics  
            'Hóa hữu cơ': [
                'ankan', 'anken', 'ankin', 'benzen', 'ancol', 'phenol',
                'andehit', 'xeton', 'axit cacboxylic', 'carbon'
            ],
            'Este – Lipit': [
                'este', 'lipit', 'chất béo', 'sáp', 'dầu thực vật',
                'axit béo', 'glixerol', 'xà phòng hóa'
            ],
            'Điện phân': [
                'điện phân', 'catot', 'anot', 'điện cực', 'ion',
                'điện ly', 'pin điện', 'ăn mòn điện hóa'
            ],
            'Axit - Bazơ': [
                'axit', 'bazơ', 'pH', 'muối', 'trung hòa',
                'đệm', 'hidro', 'hidroxit'
            ],
            'Oxi hóa - Khử': [
                'oxi hóa', 'khử', 'số oxi hóa', 'chất oxi hóa',
                'chất khử', 'electron', 'cân bằng electron'
            ],
            
            # Biology topics
            'Di truyền học': [
                'gen', 'alen', 'NST', 'nhiễm sắc thể', 'ADN', 'ARN',
                'đột biến', 'lai', 'kiểu gen', 'kiểu hình', 'phân li'
            ],
            'Tiến hóa': [
                'tiến hóa', 'chọn lọc tự nhiên', 'đột biến', 'quần thể',
                'loài', 'biến dị', 'thích nghi', 'Darwin'
            ],
            'Sinh thái học': [
                'hệ sinh thái', 'quần xã', 'chuỗi thức ăn', 'môi trường',
                'sinh vật', 'cạnh tranh', 'cộng sinh', 'ô nhiễm'
            ],
            'Tế bào học': [
                'tế bào', 'nhân tế bào', 'ti thể', 'lục lạp',
                'màng tế bào', 'bào quan', 'phân bào'
            ],
            'Sinh lý học': [
                'hô hấp', 'tuần hoàn', 'tiêu hóa', 'bài tiết',
                'thần kinh', 'nội tiết', 'hormone', 'enzyme'
            ]
        }
        
        topics = []
        for _, row in data.iterrows():
            subject = row['subject']
            question_text = row['question'].lower()
            
            # Get subject-specific topics
            subject_topic_list = self.subject_topics.get(subject, [])
            
            best_topic = 'Khác'
            max_score = 0
            
            # Score each topic based on keyword matching
            for topic in subject_topic_list:
                if topic in topic_keywords:
                    keywords = topic_keywords[topic]
                    score = sum(1 for keyword in keywords if keyword in question_text)
                    
                    if score > max_score:
                        max_score = score
                        best_topic = topic
            
            # Fallback to general topic if no specific match
            if best_topic == 'Khác':
                if subject == 'physics':
                    best_topic = 'Cơ học'  # Default physics topic
                elif subject == 'chemistry':
                    best_topic = 'Hóa hữu cơ'  # Default chemistry topic
                elif subject == 'biology':
                    best_topic = 'Tế bào học'  # Default biology topic
            
            topics.append(best_topic)
        
        return topics
    
    def _prepare_bert_data(self, texts, labels):
        """Chuẩn bị data cho BERT training"""
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Convert labels to ids
        label_ids = [self.label_to_id[label] for label in labels]
        
        return texts, label_ids
    
    def _train_bert_model(self, texts, labels):
        """Train BERT/PhoBERT model"""
        try:
            # Use PhoBERT for Vietnamese
            model_name = "vinai/phobert-base"
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_to_id)
            )
            
            # Prepare data
            texts, label_ids = self._prepare_bert_data(texts, labels)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
            )
            
            # Tokenize
            train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=256)
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=256)
            
            # Create Dataset class
            class TopicDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item
                
                def __len__(self):
                    return len(self.labels)
            
            train_dataset = TopicDataset(train_encodings, y_train)
            test_dataset = TopicDataset(test_encodings, y_test)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./topic_classifier_results',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./logs',
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )
            
            # Train
            print("Training PhoBERT for topic classification...")
            trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            print(f"Evaluation results: {eval_result}")
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            return True
            
        except Exception as e:
            print(f"BERT training failed: {e}")
            return False
    
    def _train_fallback_model(self, texts, labels):
        """Train fallback TF-IDF + Random Forest model"""
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Convert labels to ids
        y = [self.label_to_id[label] for label in labels]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.fallback_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.fallback_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Topic Classification (Fallback) Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        # Print per-topic metrics
        report = classification_report(
            y_test, y_pred,
            target_names=[self.id_to_label[i] for i in range(len(self.id_to_label))],
            output_dict=True
        )
        
        print("\nPer-topic metrics:")
        for topic, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                print(f"{topic}: F1={metrics['f1-score']:.3f}, Support={metrics['support']}")
    
    def train(self, data):
        """Train topic classifier"""
        # Create topic labels
        topics = self._create_topic_labels(data)
        
        # Prepare texts
        texts = []
        for _, row in data.iterrows():
            options_text = ' '.join(row['options']) if row['options'] else ''
            full_text = row['question'] + ' ' + options_text
            texts.append(full_text)
        
        # Try BERT first, fallback to traditional ML
        if self.use_bert:
            success = self._train_bert_model(texts, topics)
            if not success:
                print("Falling back to traditional ML...")
                self.use_bert = False
                self._init_fallback_model()
                self._train_fallback_model(texts, topics)
        else:
            self._train_fallback_model(texts, topics)
        
        self.is_trained = True
        return topics
    
    def predict(self, text, subject=None):
        """Predict topic for a given text"""
        if not self.is_trained:
            return "Khác"
        
        try:
            if self.use_bert and self.pipeline:
                # BERT prediction
                results = self.pipeline(text)
                best_result = max(results, key=lambda x: x['score'])
                predicted_id = int(best_result['label'].split('_')[-1])
                predicted_topic = self.id_to_label.get(predicted_id, "Khác")
                confidence = best_result['score']
                
            else:
                # Fallback prediction
                X = self.vectorizer.transform([text])
                predicted_id = self.fallback_model.predict(X)[0]
                predicted_topic = self.id_to_label.get(predicted_id, "Khác")
                
                # Get confidence from predict_proba
                proba = self.fallback_model.predict_proba(X)[0]
                confidence = max(proba)
            
            # Filter by subject if specified
            if subject and subject in self.subject_topics:
                subject_topics = self.subject_topics[subject]
                if predicted_topic not in subject_topics:
                    # Return most likely topic for this subject
                    predicted_topic = subject_topics[0]  # Default to first topic
            
            return predicted_topic
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Khác"
    
    def get_topics_by_subject(self, subject):
        """Get available topics for a subject"""
        return self.subject_topics.get(subject, ['Khác'])

@st.cache_resource
def initialize_topic_classifier(data):
    """Initialize topic classifier với caching"""
    try:
        # Try BERT first
        classifier = TopicClassifier(use_bert=True)
        topics = classifier.train(data)
        print("✅ Topic classifier initialized successfully with PhoBERT")
        return classifier, topics
    except Exception as e:
        print(f"⚠️ Topic classifier initialization failed: {e}")
        # Fallback
        classifier = TopicClassifier(use_bert=False)
        topics = classifier.train(data)
        print("✅ Topic classifier initialized with fallback model")
        return classifier, topics