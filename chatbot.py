import pandas as pd
import numpy as np
import json
import re
import random
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

print("Import completed successfully")


def load_vnhsge_data(data_folder='Dataset'):
    """Load dữ liệu từ thư mục Dataset"""
    subjects = ['Biology', 'Chemistry', 'Physics']
    all_data = []
    
    for subject in subjects:
        subject_path = os.path.join(data_folder, subject)
        if not os.path.exists(subject_path):
            continue
            
        json_files = glob.glob(os.path.join(subject_path, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    if 'Question' in item and 'Choice' in item:
                        question_text, options = parse_question(item['Question'])
                        
                        question_data = {
                            'id': item.get('ID', ''),
                            'question': question_text,
                            'options': options,
                            'answer': item['Choice'],
                            'subject': subject.lower(),
                            'explanation': item.get('Explanation', '')
                        }
                        all_data.append(question_data)
            except:
                continue
    
    return pd.DataFrame(all_data)

def parse_question(question_full):
    """Tách câu hỏi và đáp án"""
    lines = question_full.split('\n')
    
    question = lines[0]
    if question.startswith('Câu'):
        question = re.sub(r'^Câu \d+:\s*', '', question)
    
    options = []
    for line in lines[1:]:
        line = line.strip()
        if line and (line.startswith('A.') or line.startswith('B.') or 
                    line.startswith('C.') or line.startswith('D.')):
            options.append(line)
    
    return question.strip(), options

# Test load data
data = load_vnhsge_data()
print(f"Loaded {len(data)} questions")


def get_data_stats(data):
    """Thống kê dữ liệu"""
    stats = {}
    for subject in ['biology', 'chemistry', 'physics']:
        subject_data = data[data['subject'] == subject]
        stats[subject] = len(subject_data)
    
    return stats

stats = get_data_stats(data)
print("Data statistics:")
for subject, count in stats.items():
    print(f"{subject}: {count} questions")


def preprocess_text(text):
    """Tiền xử lý văn bản"""
    text = text.lower()
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = ' '.join(text.split())
    return text


sample_text = "Câu 1: Chất béo là gì?"
processed = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed}")


def prepare_training_data(data):
    """Chuẩn bị dữ liệu training"""
    questions = []
    subjects = []
    
    for _, row in data.iterrows():
        full_text = row['question'] + ' ' + ' '.join(row['options']) if row['options'] else row['question']
        processed_question = preprocess_text(full_text)
        questions.append(processed_question)
        subjects.append(row['subject'])
    
    return questions, subjects

questions, subjects = prepare_training_data(data)
print(f"Prepared {len(questions)} training samples")


X_train, X_test, y_train, y_test = train_test_split(
    questions, subjects, 
    test_size=0.2, 
    random_state=42, 
    stratify=subjects
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


def create_and_train_model(X_train, y_train, X_test, y_test):
    """Tạo và train model"""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    classifier = MultinomialNB(alpha=0.1)
    
    # Vectorize và train
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier.fit(X_train_vec, y_train)
    
    # Đánh giá
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return vectorizer, classifier, accuracy

vectorizer, classifier, accuracy = create_and_train_model(X_train, y_train, X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")


def save_model(vectorizer, classifier, path='vnhsge_model.pkl'):
    """Lưu model"""
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier
    }
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    return True

def load_model(path='vnhsge_model.pkl'):
    """Load model"""
    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['vectorizer'], model_data['classifier']
    except:
        return None, None


save_status = save_model(vectorizer, classifier)
print(f"Model saved: {save_status}")

def predict_subject(question, vectorizer, classifier):
    """Dự đoán môn học"""
    processed = preprocess_text(question)
    question_vec = vectorizer.transform([processed])
    prediction = classifier.predict(question_vec)[0]
    return prediction

# Test prediction
test_question = "Axit amin là đơn phân cấu tạo nên phân tử nào?"
predicted = predict_subject(test_question, vectorizer, classifier)
print(f"Predicted subject: {predicted}")

def get_random_question(data, subject=None):
    """Lấy câu hỏi ngẫu nhiên"""
    if subject:
        filtered_data = data[data['subject'] == subject]
    else:
        filtered_data = data
    
    if len(filtered_data) == 0:
        return None
    
    return filtered_data.sample(1).iloc[0]


random_question = get_random_question(data, 'biology')
print(f"Random question: {random_question['question'][:50]}...")


def check_answer(user_answer, correct_answer):
    """Kiểm tra đáp án"""
    user_answer = user_answer.upper().strip()
    correct_answer = correct_answer.upper().strip()
    
    is_correct = user_answer == correct_answer
    return is_correct

# Test answer checking
result = check_answer('A', 'A')
print(f"Answer check result: {result}")


class ScoreTracker:
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def add_result(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    
    def get_accuracy(self):
        if self.total == 0:
            return 0
        return (self.correct / self.total) * 100
    
    def reset(self):
        self.correct = 0
        self.total = 0

# Test score tracker
score_tracker = ScoreTracker()
score_tracker.add_result(True)
score_tracker.add_result(False)
print(f"Score: {score_tracker.correct}/{score_tracker.total} ({score_tracker.get_accuracy():.1f}%)")


def create_streamlit_app():
    """Tạo giao diện Streamlit"""
    
    st.title("Trắc Nghiệm")
    st.write("Hệ thống hỏi đáp trắc nghiệm Lý - Hóa - Sinh")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = load_vnhsge_data()
        st.session_state.vectorizer, st.session_state.classifier = load_model()
        st.session_state.score_tracker = ScoreTracker()
        st.session_state.current_question = None
    
    # Subject selection
    subject_options = {
        'Ngẫu nhiên': None,
        'Sinh học': 'biology',
        'Hóa học': 'chemistry', 
        'Vật lý': 'physics'
    }
    
    selected_subject = st.selectbox("Chọn môn học:", list(subject_options.keys()))
    subject_code = subject_options[selected_subject]
    
    # Get new question button
    if st.button("Câu hỏi mới"):
        st.session_state.current_question = get_random_question(st.session_state.data, subject_code)
    
    # Display current question
    if st.session_state.current_question is not None:
        question = st.session_state.current_question
        
        st.write("**Câu hỏi:**")
        st.write(question['question'])
        
        st.write("**Các đáp án:**")
        for option in question['options']:
            st.write(option)
        
        # Answer input
        user_answer = st.radio("Chọn đáp án:", ['A', 'B', 'C', 'D'])
        
        # Submit answer
        if st.button("Gửi đáp án"):
            is_correct = check_answer(user_answer, question['answer'])
            st.session_state.score_tracker.add_result(is_correct)
            
            if is_correct:
                st.success("Đúng!")
            else:
                st.error(f"Sai! Đáp án đúng là: {question['answer']}")
            
            if question['explanation']:
                st.info(f"Giải thích: {question['explanation']}")
        
        # Display score
        st.write("---")
        st.write(f"**Điểm số:** {st.session_state.score_tracker.correct}/{st.session_state.score_tracker.total}")
        st.write(f"**Độ chính xác:** {st.session_state.score_tracker.get_accuracy():.1f}%")
    
    else:
        st.write("Nhấn 'Câu hỏi mới' để bắt đầu!")


if __name__ == "__main__":
    # Để chạy app streamlit, sử dụng lệnh:
    # streamlit run filename.py
    create_streamlit_app()

print("Streamlit app ready. Run with: streamlit run filename.py")

def run_cli_version():
    """Chạy version command line"""
    print("VNHSGE - Command Line Version")
    
    data = load_vnhsge_data()
    vectorizer, classifier = load_model()
    score_tracker = ScoreTracker()
    
    subject_map = {
        '1': 'biology',
        '2': 'chemistry', 
        '3': 'physics'
    }
    
    while True:
        print("\nChọn môn: 1-Sinh, 2-Hóa, 3-Lý, 0-Thoát")
        choice = input("Lựa chọn: ").strip()
        
        if choice == '0':
            break
        
        subject = subject_map.get(choice)
        question = get_random_question(data, subject)
        
        if question is None:
            print("Không có câu hỏi!")
            continue
        
        print(f"\nCâu hỏi: {question['question']}")
        for option in question['options']:
            print(option)
        
        user_answer = input("Đáp án (A/B/C/D): ").strip()
        is_correct = check_answer(user_answer, question['answer'])
        score_tracker.add_result(is_correct)
        
        if is_correct:
            print("Đúng!")
        else:
            print(f"Sai! Đáp án đúng: {question['answer']}")
        
        if question['explanation']:
            print(f"Giải thích: {question['explanation']}")
        
        print(f"Điểm: {score_tracker.correct}/{score_tracker.total} ({score_tracker.get_accuracy():.1f}%)")

# Uncomment to run CLI version
# run_cli_version()

print("Setup completed. All functions ready to use.")