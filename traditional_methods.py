# --- 1. 手動實作 TF-IDF ---

def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)"""
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words
    return tf_dict

def calculate_idf(documents, word):
    """計算逆文件頻率 (Inverse Document Frequency)"""
    N = len(documents)
    # 計算包含該詞的文件數 (加1平滑處理避免分母為0)
    docs_with_word = sum(1 for doc in documents if word in jieba.lcut(doc))
    idf = math.log10(N / (docs_with_word + 1))
    return idf

# 執行與測試手動實作
print("=== A-1: 手動 TF-IDF 測試 ===")
sample_doc = docs_A1[0] # "人工智慧正在改變世界..."
words = jieba.lcut(sample_doc)
word_counts = Counter(words)
tf_result = calculate_tf(word_counts, len(words))
idf_result = calculate_idf(docs_A1, "人工智慧")

print(f"測試文件: {sample_doc[:10]}...")
print(f"詞彙 '人工智慧' 的 TF 值: {tf_result.get('人工智慧', 0):.4f}")
print(f"詞彙 '人工智慧' 的 IDF 值: {idf_result:.4f}")


#這邊就只是在把詞切出來以後，計算他在一段話中的單字占比/計算在一篇文章內某個詞的出現比例

# --- 2. 使用 scikit-learn 實作與計算相似度 [cite: 50-52] ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 為了讓 sklearn 處理中文，需要先用空白連接分詞結果
docs_segmented = [" ".join(jieba.lcut(doc)) for doc in docs_A1]
# 把詞切分以後中間插入空白鍵

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_segmented)
similarity_matrix = cosine_similarity(tfidf_matrix)

print("\n=== A-1: Sklearn 相似度矩陣 (前3句) ===")
print(similarity_matrix[:3, :3])
print("(數值越高代表越相似)")


#Part A-2: 基於規則的文本分類

class RuleBasedSentimentClassifier:
    def __init__(self):
        # [cite: 67-72]
        self.positive_words = ['好','棒','優秀','喜歡','推薦','滿意','開心','值得','精彩','完美']
        self.negative_words = ['差','糟','失望','討厭','不推薦','浪費','無聊','爛','糟糕','差勁']
        self.negation_words = ['不','沒','無','非','別']

    def classify(self, text):
        words = jieba.lcut(text)
        score = 0
        
        # 簡易邏輯：檢查每個詞，若前面有否定詞則反轉分數
        for i, word in enumerate(words):
            word_score = 0
            if word in self.positive_words:
                word_score = 1
            elif word in self.negative_words:
                word_score = -1
            
            # 檢查前一個詞是否為否定詞
            if i > 0 and words[i-1] in self.negation_words:
                word_score *= -1
            
            score += word_score
            
        if score > 0: return "正面"
        elif score < 0: return "負面"
        else: return "中性"

class TopicClassifier:
    def __init__(self):
        # [cite: 84-93]
        self.topic_keywords = {
            '科技': ['AI','人工智慧','電腦','軟體','程式','演算法','深度學習','機器學習'],
            '運動': ['運動','健身','跑步','游泳','球類','比賽','慢跑','重訓'],
            '美食': ['吃','食物','餐廳','美味','料理','烹飪','牛肉麵','好吃'],
            '旅遊': ['旅行','景點','飯店','機票','觀光','度假']
        }

    def classify(self, text):
        counts = {topic: 0 for topic in self.topic_keywords}
        words = jieba.lcut(text)
        
        for word in words:
            for topic, keywords in self.topic_keywords.items():
                if word in keywords:
                    counts[topic] += 1
        
        # 找出分數最高的主題
        return max(counts, key=counts.get)

# 執行與測試 
print("\n=== A-2: 規則分類器測試 ===")
sentiment_clf = RuleBasedSentimentClassifier()
topic_clf = TopicClassifier()

for text in test_texts_A2:
    s_result = sentiment_clf.classify(text)
    t_result = topic_clf.classify(text)
    print(f"文本: {text[:15]}...")
    print(f" -> 情感: {s_result}, 主題: {t_result}")

#Part A-3 : 統計式自動摘要

class StatisticalSummarizer:
    def __init__(self):
        # [cite: 113-117] 簡易停用詞
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '也', '很', '人', '到', '都', '一個', '上', '說', '要', '去', '你們', '我們'])

    def sentence_score(self, sentence, word_freq):
        """計算句子重要性分數"""
        words = jieba.lcut(sentence)
        if len(words) == 0: return 0
        
        # 1. 高頻詞分數
        score = sum(word_freq.get(w, 0) for w in words if w not in self.stop_words)
        
        # 2. 句子長度懲罰 (過短不具代表性)
        if len(words) < 5: score *= 0.5
        
        return score

    def summarize(self, text, ratio=0.3):
        # 1. 分句 (簡單以句號逗號換行分割)
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 2. 計算全文詞頻
        all_words = jieba.lcut(text)
        word_freq = Counter([w for w in all_words if w not in self.stop_words])
        
        # 3. 計算句構分數
        scores = []
        for i, sent in enumerate(sentences):
            base_score = self.sentence_score(sent, word_freq)
            # 位置加權 (首句通常重要)
            if i == 0: base_score *= 1.2
            scores.append((i, sent, base_score))
            
        # 4. 選擇最高分的句子
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(scores, key=lambda x: x[2], reverse=True)[:num_sentences]
        
        # 5. 按原文順序排列
        top_sentences.sort(key=lambda x: x[0])
        
        return "。".join([s[1] for s in top_sentences]) + "。"

# 執行與測試
print("\n=== A-3: 統計摘要測試 ===")
summarizer = StatisticalSummarizer()
summary = summarizer.summarize(article_A3, ratio=0.3)
print(f"原文長度: {len(article_A3)}")
print(f"摘要結果:\n{summary}")




