"""
main.py - Streamlit App Main File

LOGIC CHÍNH:
1. Cache dữ liệu và models để tối ưu performance
2. Giao diện filter: môn học, năm, độ khó, CHƯƠNG (topic)
3. Hiển thị câu hỏi ngẫu nhiên và xử lý đáp án
4. Hiển thị câu hỏi tương tự sau khi trả lời
5. Theo dõi điểm số người dùng

FLOW:
- Load data → Initialize models → Initialize topic classifier → Create filters → Display question → Check answer → Show similar

NEW FEATURE: Topic Classification
- Phân loại chủ đề câu hỏi theo chương sử dụng BERT/PhoBERT
- Filter theo chương: VD Physics → "Dao động cơ", "Điện xoay chiều"
"""

import streamlit as st
from utils import load_vnhsge_data, get_random_question, check_answer, ScoreTracker
from models import initialize_models
from topic_classifier import initialize_topic_classifier
import warnings
warnings.filterwarnings('ignore')

def create_streamlit_app():
    """Tạo giao diện Streamlit"""
    
    st.title("Hệ thống trắc nghiệm")
    st.write("VNHSGE đề thi THPT Quốc gia gồm 3 môn (Lý-Hóa-Sinh) (2019~2023)")
    
    # Initialize session state - chỉ load một lần
    if 'data' not in st.session_state:
        raw_data = load_vnhsge_data()
        
        # Initialize difficulty classifier và similar finder
        difficulty_classifier, difficulties, similar_finder = initialize_models(raw_data)
        
        # Initialize topic classifier
        topic_classifier, topics = initialize_topic_classifier(raw_data)
        
        # Thêm difficulty và topic columns vào data
        data_enhanced = raw_data.copy()
        data_enhanced['difficulty'] = difficulties
        data_enhanced['topic'] = topics
        
        st.session_state.data = data_enhanced
        st.session_state.similar_finder = similar_finder
        st.session_state.topic_classifier = topic_classifier
        st.session_state.score_tracker = ScoreTracker()
        st.session_state.current_question = None
        st.session_state.show_similar = False
    
    # Filter options
    subject_options = {
        'Sinh học': 'biology',
        'Hóa học': 'chemistry', 
        'Vật lý': 'physics'
    }
    
    year_options = {
        'Tất cả các năm': None,
        '2019': 2019, '2020': 2020, '2021': 2021, '2022': 2022, '2023': 2023
    }
    
    difficulty_options = {
        'Tất cả mức độ': None,
        'Dễ': 'easy', 'Trung bình': 'medium', 'Khó': 'hard'
    }
    
    # UI Filters
    selected_subject = st.selectbox("Chọn môn học:", list(subject_options.keys()))
    selected_year = st.selectbox("Chọn đề theo năm:", list(year_options.keys()))
    selected_difficulty = st.selectbox("Chọn mức độ câu hỏi:", list(difficulty_options.keys()))
    
    # Get filter values
    subject_code = subject_options[selected_subject]
    year_code = year_options[selected_year]
    difficulty_code = difficulty_options[selected_difficulty]
    
    # Topic filter (NEW)
    available_topics = st.session_state.topic_classifier.get_topics_by_subject(subject_code)
    topic_options = ['Tất cả chương'] + available_topics
    selected_topic = st.selectbox("Chọn chương:", topic_options)
    topic_code = None if selected_topic == 'Tất cả chương' else selected_topic
    
    # New question button
    if st.button("Câu hỏi mới"):
        st.session_state.current_question = get_random_question(
            st.session_state.data, subject_code, year_code, difficulty_code, topic_code
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
                explanation = question['explanation'].replace('\\n', '\n')
                st.info(f"Giải thích:\n{explanation}")
            
            st.session_state.show_similar = True
        
        # Similar questions
        if st.session_state.show_similar:
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
                        
                        if st.button(f"Làm bài này", key=f"do_question_{i}"):
                            st.session_state.current_question = similar_q
                            st.session_state.show_similar = False
                            st.rerun()
        
        # Score display
        st.write("---")
        st.write(f"**Điểm số:** {st.session_state.score_tracker.correct}/{st.session_state.score_tracker.total}")
        st.write(f"**Độ chính xác:** {st.session_state.score_tracker.get_accuracy():.1f}%")
    
    else:
        st.write("Nhấn 'Câu hỏi mới' để bắt đầu!")

if __name__ == "__main__":
    create_streamlit_app()