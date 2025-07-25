"""
utils.py - Utility Functions Module

LOGIC CHÍNH:
1. Data loading: Load và parse dữ liệu từ JSON files
2. Question handling: Random selection, filtering theo filters
3. Answer checking: So sánh đáp án người dùng vs đúng
4. Score tracking: Theo dõi correct/total và accuracy
5. LaTeX conversion: Convert LaTeX formulas thành plain text

OPTIMIZATION:
- Streamlit caching cho data loading
- Simplified LaTeX conversion
- Removed duplicate text processing
"""

import pandas as pd
import json
import re
import os
import glob
import streamlit as st

@st.cache_data
def convert_latex_to_text(text):
    """Convert LaTeX formulas sang plain text"""
    if not isinstance(text, str):
        return text
    
    # Convert \n to actual line breaks
    text = text.replace('\\n', '\n')
    
    # LaTeX replacements
    replacements = {
        # Fractions
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        
        # Superscripts and subscripts
        r'\^{([^}]+)}': r'^(\1)',
        r'_{([^}]+)}': r'_(\1)',
        
        # Arrows
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        
        # Greek letters
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\pi': 'π', r'\\sigma': 'σ',
        
        # Math symbols
        r'\\times': '×', r'\\div': '÷', r'\\pm': '±',
        r'\\leq': '≤', r'\\geq': '≥', r'\\neq': '≠',
        
        # Clean up
        r'\{([^}]+)\}': r'\1',  # Remove braces
        r'\\text\{([^}]+)\}': r'\1',  # Remove text command
        r'\\left': '', r'\\right': '',  # Remove left/right
        r'\s+': ' ',  # Multiple spaces to single
    }
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

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
                        
                        question_data = {
                            'id': item.get('ID', ''),
                            'question': question_text,
                            'options': options,
                            'answer': item['Choice'],
                            'subject': subject.lower(),
                            'explanation': convert_latex_to_text(item.get('Explanation', ''))
                        }
                        all_data.append(question_data)
            except:
                continue
    
    return pd.DataFrame(all_data)

def parse_question(question_full):
    """Tách câu hỏi và đáp án"""
    lines = question_full.split('\n')
    
    # Extract question (remove "Câu X:" prefix)
    question = lines[0]
    if question.startswith('Câu'):
        question = re.sub(r'^Câu \d+:\s*', '', question)
    
    # Extract options
    options = []
    for line in lines[1:]:
        line = line.strip()
        if line and line.startswith(('A.', 'B.', 'C.', 'D.')):
            options.append(line)
    
    return question.strip(), options

def get_random_question(data, subject=None, year=None, difficulty=None, topic=None):
    """Lấy câu hỏi ngẫu nhiên theo filters (bao gồm topic mới)"""
    filtered_data = data.copy()
    
    # Apply filters
    if subject:
        filtered_data = filtered_data[filtered_data['subject'] == subject]
    
    if year:
        filtered_data = filtered_data[filtered_data['id'].str.contains(str(year), na=False)]
    
    if difficulty and 'difficulty' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['difficulty'] == difficulty]
    
    # NEW: Topic filter
    if topic and 'topic' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['topic'] == topic]
    
    if len(filtered_data) == 0:
        return None
    
    return filtered_data.sample(1).iloc[0]

def check_answer(user_answer, correct_answer):
    """Kiểm tra đáp án"""
    return user_answer.upper().strip() == correct_answer.upper().strip()

class ScoreTracker:
    """Theo dõi điểm số người dùng"""
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def add_result(self, is_correct):
        """Thêm kết quả trả lời"""
        self.total += 1
        if is_correct:
            self.correct += 1
    
    def get_accuracy(self):
        """Tính độ chính xác"""
        return (self.correct / self.total * 100) if self.total > 0 else 0