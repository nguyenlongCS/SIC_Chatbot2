"""
utils.py - Utility Functions Module

Chứa các helper functions và utilities:
1. Data loading: Load và parse dữ liệu từ JSON files
2. Question handling: Random selection, filtering theo môn/năm/độ khó
3. Answer checking: Kiểm tra đáp án người dùng
4. Score tracking: Theo dõi điểm số và độ chính xác
5. Text preprocessing: Wrapper cho Vietnamese NLP
6. LaTeX conversion: Convert LaTeX formulas sang plain text

Tất cả functions được tối ưu với Streamlit caching.
"""

import pandas as pd
import json
import re
import os
import glob
import streamlit as st

@st.cache_data
def convert_latex_to_text(text):
    """Convert LaTeX formulas sang plain text dễ đọc"""
    if not isinstance(text, str):
        return text
    
    # Convert \n to actual line breaks first
    text = text.replace('\\n', '\n')
    
    # Replace common LaTeX commands
    replacements = {
        # Fractions
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        r'\\frac([^{}\s])([^{}\s])': r'(\1)/(\2)',
        
        # Superscripts
        r'\^{([^}]+)}': r'^(\1)',
        r'\^([0-9])': r'^(\1)',
        
        # Subscripts  
        r'_{([^}]+)}': r'_(\1)',
        r'_([0-9])': r'_(\1)',
        
        # Arrows
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        r'\\Rightarrow': '⇒',
        
        # Greek letters
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\pi': 'π',
        r'\\sigma': 'σ',
        
        # Math symbols
        r'\\times': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\mp': '∓',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\infty': '∞',
        
        # Remove extra LaTeX formatting
        r'\{([^}]+)\}': r'\1',  # Remove braces
        r'\\text\{([^}]+)\}': r'\1',  # Remove text command
        r'\\mathrm\{([^}]+)\}': r'\1',  # Remove mathrm
        r'\\left': '',  # Remove left
        r'\\right': '',  # Remove right
        
        # Clean up spaces
        r'\s+': ' ',  # Multiple spaces to single
    }
    
    # Apply replacements
    converted_text = text
    for pattern, replacement in replacements.items():
        converted_text = re.sub(pattern, replacement, converted_text)
    
    # Additional cleaning for specific patterns
    converted_text = re.sub(r'([A-Za-z])\*([A-Za-z])', r'\1_\2', converted_text)  # p*A → p_A
    converted_text = re.sub(r'F_(\d+)', r'F\1', converted_text)  # F_{1} → F1
    converted_text = re.sub(r'1/2\^(\d+)', r'(1/2)^\1', converted_text)  # 1/2^3 → (1/2)^3
    
    return converted_text.strip()

@st.cache_data
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
                        
                        # Convert LaTeX in explanation
                        explanation = item.get('Explanation', '')
                        explanation = convert_latex_to_text(explanation)
                        
                        question_data = {
                            'id': item.get('ID', ''),
                            'question': question_text,
                            'options': options,
                            'answer': item['Choice'],
                            'subject': subject.lower(),
                            'explanation': explanation
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

def get_random_question(data, subject=None, year=None, difficulty=None):
    """Lấy câu hỏi ngẫu nhiên"""
    filtered_data = data.copy()
    
    if subject:
        filtered_data = filtered_data[filtered_data['subject'] == subject]
    
    if year:
        filtered_data = filtered_data[filtered_data['id'].str.contains(str(year), na=False)]
    
    if difficulty and 'difficulty' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['difficulty'] == difficulty]
    
    if len(filtered_data) == 0:
        return None
    
    return filtered_data.sample(1).iloc[0]

def check_answer(user_answer, correct_answer):
    """Kiểm tra đáp án"""
    user_answer = user_answer.upper().strip()
    correct_answer = correct_answer.upper().strip()
    return user_answer == correct_answer

class ScoreTracker:
    """Theo dõi điểm số người dùng"""
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