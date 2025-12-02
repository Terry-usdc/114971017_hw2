#Part B: 現代 AI 方法 (使用 gpt-5-nano

# --- B-1: 語意相似度計算---
def ai_similarity(text1, text2):
    prompt = f"""
    請評估以下兩段文字的語意相似度。
    文字1: {text1}
    文字2: {text2}
    請只回答一個0-100的數字,代表相似度百分比,不要有其他文字。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",  #gpt-5-nano當模型
            messages=[{"role": "user", "content": prompt}]

        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- B-2: AI 文本分類 [cite: 168-178] ---
def ai_classify(text):
    prompt = f"""
    請對以下文本進行分類，返回 JSON 格式:
    {{
        "sentiment": "正面/負面/中性",
        "topic": "主題類別(如科技/運動/美食/旅遊)",
        "confidence": 0.0到1.0之間的數值
    }}
    文本: {text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # 確保返回 JSON

        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- B-3: AI 自動摘要 [cite: 179-188] ---
def ai_summarize(text, max_length=100):
    prompt = f"""
    請幫我摘要以下文章，長度控制在 {max_length} 字以內，保留關鍵資訊並確保語句通順。
    文章: {text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],

        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# === Part B 執行測試 ===
print("\n=== B-1: AI 相似度測試 ===")
sim_score = ai_similarity(docs_A1[0], docs_A1[3]) # 比較兩句關於AI的句子(第0跟第3句)，確實只return了'85'
print(f"句子1: {docs_A1[0][:10]}...\n句子2: {docs_A1[3][:10]}...\n相似度分數: {sim_score}")

print("\n=== B-2: AI 分類測試 ===")
cls_result = ai_classify(test_texts_A2[0]) # 牛肉麵那句，回傳了{\n  "sentiment": "正面",\n  "topic": "美食",\n  "confidence": 0.93\n}
print(f"文本: {test_texts_A2[0][:10]}...\n分類結果: {cls_result}")

print("\n=== B-3: AI 摘要測試 ===")
ai_summary_result = ai_summarize(article_A3, max_length=100)
print(f"AI 摘要結果:\n{ai_summary_result}")
