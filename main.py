"""
main.py - Streamlit App Main File

Giao diện chính của hệ thống trắc nghiệm VNHSGE:
1. Load dữ liệu và khởi tạo models (với caching)
2. Giao diện chọn: môn học, năm, độ khó
3. Hiển thị câu hỏi và xử lý đáp án
4. Hiển thị câu hỏi tương tự sau khi trả lời
5. Theo dõi điểm số người dùng

Sử dụng session state để quản lý trạng thái ứng dụng.
Tích hợp Vietnamese NLP để xử lý văn bản tiếng Việt.
"""

import streamlit as st
from utils import load_vnhsge_data, get_random_question, check_answer, ScoreTracker
from models import initialize_models
from vietnamese_nlp import clean_dataset
import warnings
warnings.filterwarnings('ignore')

def create_streamlit_app():
    """Tạo giao diện Streamlit"""
    
    st.title("Trắc Nghiệm")
    st.write("Hệ thống hỏi đáp trắc nghiệm Lý - Hóa - Sinh")
    
    # Initialize session state with caching
    if 'data' not in st.session_state:
        # Load raw data
        raw_data = load_vnhsge_data()
        
        # Use cached model initialization (this will add difficulty column)
        difficulty_classifier, difficulties, similar_finder = initialize_models(raw_data)
        
        # Now clean the data with difficulty column included
        data_with_difficulty = raw_data.copy()
        data_with_difficulty['difficulty'] = difficulties
        
        # Simple text cleaning without balancing (since balancing causes issues)
        from vietnamese_nlp import VietnameseNLP
        nlp = VietnameseNLP()
        
        # Clean text content only
        data_with_difficulty['question'] = data_with_difficulty['question'].apply(
            lambda x: nlp.clean_text_advanced(x, remove_stopwords=True, normalize=True, stem=False)
        )
        
        # Remove empty texts
        data_with_difficulty = data_with_difficulty[data_with_difficulty['question'].str.strip() != ''].reset_index(drop=True)
        
        st.session_state.data = data_with_difficulty
        st.session_state.similar_finder = similar_finder
        
        st.session_state.score_tracker = ScoreTracker()
        st.session_state.current_question = None
        st.session_state.show_similar = False
    
    # Subject selection
    subject_options = {
        'Sinh học': 'biology',
        'Hóa học': 'chemistry', 
        'Vật lý': 'physics'
    }
    
    selected_subject = st.selectbox("Chọn môn học:", list(subject_options.keys()))
    subject_code = subject_options[selected_subject]
    
    # Year selection
    year_options = {
        'Tất cả các năm': None,
        '2019': 2019,
        '2020': 2020,
        '2021': 2021,
        '2022': 2022,
        '2023': 2023
    }
    
    selected_year = st.selectbox("Chọn đề theo năm:", list(year_options.keys()))
    year_code = year_options[selected_year]
    
    # Difficulty selection
    difficulty_options = {
        'Tất cả mức độ': None,
        'Dễ': 'easy',
        'Trung bình': 'medium',
        'Khó': 'hard'
    }
    
    selected_difficulty = st.selectbox("Chọn mức độ câu hỏi:", list(difficulty_options.keys()))
    difficulty_code = difficulty_options[selected_difficulty]
    
    # Get new question button
    if st.button("Câu hỏi mới"):
        st.session_state.current_question = get_random_question(
            st.session_state.data, subject_code, year_code, difficulty_code
        )
        st.session_state.show_similar = False
    
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
                # Convert explanation and display with proper line breaks
                explanation = question['explanation'].replace('\\n', '\n')
                st.info(f"Giải thích:\n{explanation}")
            
            # Show similar questions after answering
            st.session_state.show_similar = True
        
        # Display similar questions if answer was submitted
        if st.session_state.show_similar:
            # Use cached similar finder
            similar_questions = st.session_state.similar_finder.find_similar_questions(
                question['id'], n_similar=3
            )
            
            if similar_questions:
                st.write("---")
                st.write("**💡 Câu hỏi tương tự để luyện tập:**")
                
                for i, similar in enumerate(similar_questions, 1):
                    similar_q = similar['question_data']
                    similarity_score = similar['similarity']
                    
                    with st.expander(f"Câu {i}: {similar_q['question'][:60]}... (Độ tương đồng: {similarity_score:.2f})"):
                        st.write("**Câu hỏi:**")
                        st.write(similar_q['question'])
                        
                        st.write("**Các đáp án:**")
                        for option in similar_q['options']:
                            st.write(option)
                        
                        # Add "Làm bài này" button
                        if st.button(f"Làm bài này", key=f"do_question_{i}"):
                            st.session_state.current_question = similar_q
                            st.session_state.show_similar = False
                            st.rerun()
        
        # Display score
        st.write("---")
        st.write(f"**Điểm số:** {st.session_state.score_tracker.correct}/{st.session_state.score_tracker.total}")
        st.write(f"**Độ chính xác:** {st.session_state.score_tracker.get_accuracy():.1f}%")
    
    else:
        st.write("Nhấn 'Câu hỏi mới' để bắt đầu!")

if __name__ == "__main__":
    create_streamlit_app()  