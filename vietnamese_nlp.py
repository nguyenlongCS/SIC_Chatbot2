import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
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
        
        # Vietnamese accent mappings for normalization
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
        
        # Common Vietnamese word endings for simple stemming
        self.suffixes = ['tion', 'sion', 'ness', 'ment', 'ing', 'ed', 'er', 'est', 'ly', 's', 'es']
        
    def normalize_vietnamese(self, text, remove_accents=False):
        """Chuẩn hóa văn bản tiếng Việt"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove accents if specified
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
    
    def simple_vietnamese_stem(self, word):
        """Stemming đơn giản cho tiếng Việt"""
        if not isinstance(word, str) or len(word) < 4:
            return word
        
        # Simple rule-based stemming
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def stem_text(self, text):
        """Stemming cho toàn bộ văn bản"""
        if not isinstance(text, str):
            return ""
        
        words = text.split()
        stemmed_words = [self.simple_vietnamese_stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def clean_text_advanced(self, text, remove_stopwords=True, normalize=True, stem=False):
        """Làm sạch văn bản tiếng Việt nâng cao"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Vietnamese
        if normalize:
            text = self.normalize_vietnamese(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Stemming
        if stem:
            text = self.stem_text(text)
        
        return text
    
    def detect_duplicates(self, data, text_column='question', similarity_threshold=0.9):
        """Phát hiện câu hỏi trùng lặp"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Clean texts for comparison
        cleaned_texts = [self.clean_text_advanced(text, remove_stopwords=True, normalize=True) 
                        for text in data[text_column]]
        
        # Vectorize
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        text_vectors = vectorizer.fit_transform(cleaned_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(text_vectors)
        
        # Find duplicates
        duplicates = []
        n = len(data)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i][j] > similarity_threshold:
                    duplicates.append({
                        'index1': i,
                        'index2': j,
                        'similarity': similarities[i][j],
                        'text1': data.iloc[i][text_column][:50] + "...",
                        'text2': data.iloc[j][text_column][:50] + "..."
                    })
        
        print(f"Found {len(duplicates)} potential duplicates")
        return duplicates
    
    def remove_duplicates(self, data, text_column='question', similarity_threshold=0.9):
        """Loại bỏ câu hỏi trùng lặp"""
        duplicates = self.detect_duplicates(data, text_column, similarity_threshold)
        
        # Get indices to remove (keep first occurrence)
        indices_to_remove = set()
        for dup in duplicates:
            indices_to_remove.add(dup['index2'])
        
        # Remove duplicates
        cleaned_data = data.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        print(f"Removed {len(indices_to_remove)} duplicate questions")
        print(f"Original: {len(data)} questions -> Cleaned: {len(cleaned_data)} questions")
        
        return cleaned_data
    
    def detect_outliers(self, data, text_column='question'):
        """Phát hiện outliers dựa trên độ dài văn bản"""
        lengths = [len(str(text).split()) for text in data[text_column]]
        
        Q1 = np.percentile(lengths, 25)
        Q3 = np.percentile(lengths, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, length in enumerate(lengths):
            if length < lower_bound or length > upper_bound:
                outliers.append({
                    'index': i,
                    'length': length,
                    'text': str(data.iloc[i][text_column])[:100] + "...",
                    'type': 'too_short' if length < lower_bound else 'too_long'
                })
        
        print(f"Found {len(outliers)} outliers:")
        print(f"- Too short: {len([o for o in outliers if o['type'] == 'too_short'])}")
        print(f"- Too long: {len([o for o in outliers if o['type'] == 'too_long'])}")
        
        return outliers
    
    def remove_outliers(self, data, text_column='question', remove_short=True, remove_long=True):
        """Loại bỏ outliers"""
        outliers = self.detect_outliers(data, text_column)
        
        indices_to_remove = set()
        for outlier in outliers:
            if (remove_short and outlier['type'] == 'too_short') or \
               (remove_long and outlier['type'] == 'too_long'):
                indices_to_remove.add(outlier['index'])
        
        cleaned_data = data.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        print(f"Removed {len(indices_to_remove)} outlier questions")
        return cleaned_data
    
    def balance_dataset(self, data, target_column='difficulty', strategy='oversample'):
        """Xử lý imbalanced dataset"""
        # Check current distribution
        class_counts = data[target_column].value_counts()
        print("Original class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(data)*100:.1f}%)")
        
        # Check if balancing is needed
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio < 2.0:
            print(f"Dataset is balanced (ratio: {imbalance_ratio:.2f}). No balancing needed.")
            return data
        
        print(f"Imbalance detected (ratio: {imbalance_ratio:.2f}). Applying {strategy}...")
        
        if strategy == 'oversample':
            return self._oversample(data, target_column)
        elif strategy == 'undersample':
            return self._undersample(data, target_column)
        else:
            print("Unknown strategy. Returning original data.")
            return data
    
    def _oversample(self, data, target_column):
        """Oversample minority classes"""
        class_counts = data[target_column].value_counts()
        max_count = class_counts.max()
        
        balanced_data = []
        for class_name in class_counts.index:
            class_data = data[data[target_column] == class_name]
            
            if len(class_data) < max_count:
                # Oversample
                oversampled = resample(class_data, 
                                     replace=True, 
                                     n_samples=max_count, 
                                     random_state=42)
                balanced_data.append(oversampled)
            else:
                balanced_data.append(class_data)
        
        result = pd.concat(balanced_data).reset_index(drop=True)
        
        # Shuffle the result
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("After oversampling:")
        new_counts = result[target_column].value_counts()
        for class_name, count in new_counts.items():
            print(f"  {class_name}: {count} ({count/len(result)*100:.1f}%)")
        
        return result
    
    def _undersample(self, data, target_column):
        """Undersample majority classes"""
        class_counts = data[target_column].value_counts()
        min_count = class_counts.min()
        
        balanced_data = []
        for class_name in class_counts.index:
            class_data = data[data[target_column] == class_name]
            
            if len(class_data) > min_count:
                # Undersample
                undersampled = resample(class_data, 
                                      replace=False, 
                                      n_samples=min_count, 
                                      random_state=42)
                balanced_data.append(undersampled)
            else:
                balanced_data.append(class_data)
        
        result = pd.concat(balanced_data).reset_index(drop=True)
        
        # Shuffle the result
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("After undersampling:")
        new_counts = result[target_column].value_counts()
        for class_name, count in new_counts.items():
            print(f"  {class_name}: {count} ({count/len(result)*100:.1f}%)")
        
        return result
    
    def comprehensive_clean(self, data, text_column='question', target_column='difficulty'):
        """Làm sạch dữ liệu toàn diện"""
        print("Starting comprehensive data cleaning...")
        print(f"Original dataset: {len(data)} samples")
        
        # 1. Clean text content
        print("\n1. Cleaning text content...")
        data[text_column] = data[text_column].apply(
            lambda x: self.clean_text_advanced(x, remove_stopwords=True, normalize=True, stem=False)
        )
        
        # 2. Remove empty texts
        original_len = len(data)
        data = data[data[text_column].str.strip() != ''].reset_index(drop=True)
        print(f"Removed {original_len - len(data)} empty texts")
        
        # 3. Remove duplicates
        print("\n2. Removing duplicates...")
        data = self.remove_duplicates(data, text_column)
        
        # 4. Remove outliers
        print("\n3. Removing outliers...")
        data = self.remove_outliers(data, text_column, remove_short=True, remove_long=False)
        
        # 5. Balance dataset if needed
        print("\n4. Checking dataset balance...")
        data = self.balance_dataset(data, target_column, strategy='oversample')
        
        print(f"\nFinal dataset: {len(data)} samples")
        print("Data cleaning completed!")
        
        return data

# Convenience functions for easy import
def clean_vietnamese_text(text, remove_stopwords=True, normalize=True, stem=False):
    """Quick text cleaning function"""
    nlp = VietnameseNLP()
    return nlp.clean_text_advanced(text, remove_stopwords, normalize, stem)

def clean_dataset(data, text_column='question', target_column='difficulty'):
    """Quick dataset cleaning function"""
    nlp = VietnameseNLP()
    return nlp.comprehensive_clean(data, text_column, target_column)

# Example usage
if __name__ == "__main__":
    # Test the functions
    sample_text = "Câu hỏi này là về hóa học và có nhiều từ không cần thiết"
    nlp = VietnameseNLP()
    
    print("Original:", sample_text)
    print("Cleaned:", nlp.clean_text_advanced(sample_text))
    print("No stopwords:", nlp.clean_text_advanced(sample_text, remove_stopwords=True))
    print("Normalized:", nlp.clean_text_advanced(sample_text, normalize=True, remove_stopwords=True))