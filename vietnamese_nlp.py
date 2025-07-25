"""
vietnamese_nlp.py - Vietnamese NLP Processing Module

LOGIC CHÍNH:
1. VietnameseNLP class: Xử lý văn bản tiếng Việt
   - Text normalization (lowercase, remove accents)
   - Stopwords removal
   - Simple stemming
   - Advanced text cleaning

2. Convenience functions: clean_vietnamese_text()

OPTIMIZATION:
- Removed unused features (duplicate detection, balancing, outliers)
- Kept only essential text processing
- Simplified class structure
"""

import re
import warnings
warnings.filterwarnings('ignore')

class VietnameseNLP:
    def __init__(self):
        # Vietnamese stopwords
        self.stopwords = {
            'và', 'của', 'có', 'là', 'trong', 'với', 'được', 'cho', 'từ', 'các', 'một', 'những',
            'này', 'đó', 'khi', 'để', 'không', 'về', 'sau', 'trước', 'hay', 'hoặc', 'nếu', 'như',
            'đã', 'sẽ', 'có thể', 'phải', 'nên', 'cần', 'bằng', 'theo', 'nhưng', 'mà', 'vì', 'do',
            'tại', 'trên', 'dưới', 'bên', 'giữa', 'ngoài', 'trong', 'nước', 'người', 'thời', 'lúc',
            'chỉ', 'rất', 'nhiều', 'ít', 'lại', 'thêm', 'cùng', 'cũng', 'đều', 'gì', 'ai', 'đâu',
            'sao', 'thế', 'nào', 'bao', 'mấy', 'bé', 'lớn', 'to', 'nhỏ', 'cao', 'thấp', 'xa', 'gần'
        }
        
        # Vietnamese accent mappings
        self.accent_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd'
        }
    
    def normalize_vietnamese(self, text, remove_accents=False):
        """Chuẩn hóa văn bản tiếng Việt"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        if remove_accents:
            for accented, base in self.accent_map.items():
                text = text.replace(accented, base)
        
        return text
    
    def remove_stopwords(self, text):
        """Loại bỏ stopwords tiếng Việt"""
        if not isinstance(text, str):
            return ""
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def clean_text_advanced(self, text, remove_stopwords=True, normalize=True, stem=False):
        """Làm sạch văn bản tiếng Việt nâng cao"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning - keep Vietnamese characters
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Vietnamese
        if normalize:
            text = self.normalize_vietnamese(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text

# Convenience function for easy import
def clean_vietnamese_text(text, remove_stopwords=True, normalize=True, stem=False):
    """Quick text cleaning function"""
    nlp = VietnameseNLP()
    return nlp.clean_text_advanced(text, remove_stopwords, normalize, stem)