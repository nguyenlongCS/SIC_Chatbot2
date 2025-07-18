import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Chatbot",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_vnhsge_data():
    """Load dữ liệu từ thư mục Dataset với error handling"""
    subjects = ['Biology', 'Chemistry', 'Physics']
    all_data = []
    
    # Get current directory
    current_dir = os.path.dirname(__file__) if __file__ else '.'
    data_folder = os.path.join(current_dir, 'Dataset')
    
    if not os.path.exists(data_folder):
        st.error(f"Không tìm thấy thư mục Dataset tại: {data_folder}")
        return pd.DataFrame()
    
    for subject in subjects:
        subject_path = os.path.join(data_folder, subject)
        if not os.path.exists(subject_path):
            st.warning(f"Không tìm thấy thư mục {subject}")
            continue
            
        json_files = glob.glob(os.path.join(subject_path, "*.json"))
        
        if not json_files:
            st.warning(f"Không có file JSON trong {subject}")
            continue
        
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
            except Exception as e:
                st.error(f"Lỗi đọc file {json_file}: {e}")
                continue
    
    if not all_data:
        st.error("Không thể load được dữ liệu nào!")
        return pd.DataFrame()
    
    return pd.DataFrame(all_data)

# Your existing functions with error handling...

def main():
    """Main app function"""
    st.title("🎓 Chatbot Trắc Nghiệm VNHSGE")
    st.markdown("*Hệ thống ôn tập trắc nghiệm Lý - Hóa - Sinh*")
    
    # Load data with error handling
    try:
        data = load_vnhsge_data()
        if data.empty:
            st.error("Không có dữ liệu để hiển thị!")
            return
            
        st.success(f"✅ Đã load {len(data)} câu hỏi")
        
        # Your existing app logic...
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo ứng dụng: {e}")
        st.info("Vui lòng kiểm tra lại cấu trúc thư mục và dữ liệu.")

if __name__ == "__main__":
    main()