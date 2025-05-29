import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# Load model and tokenizer
@st.cache_resource
def load_model():
    
    model = BertForSequenceClassification.from_pretrained("sentiment_model_bert")
    tokenizer = BertTokenizer.from_pretrained("sentiment_model_bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Dự đoán cảm xúc cho một đoạn văn hoặc mệnh đề
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction  # 0: Tệ, 1: Trung tính, 2: Tốt

# Chia câu thành các mệnh đề
def split_clauses(sentence):
    clauses = re.split(r'[,.;!?]', sentence)
    return [clause.strip() for clause in clauses if clause.strip()]

# Dự đoán theo từng mệnh đề và tính trung bình
def score_sentence(sentence):
    clauses = split_clauses(sentence)
    if not clauses:
        return [], None
    scores = [predict_sentiment(clause) for clause in clauses]
    avg_score = sum(scores) / len(scores)
    return scores, avg_score

# Giao diện Streamlit
st.title("Phân tích cảm xúc câu tiếng Việt")

sentence = st.text_area("Nhập câu hoặc đoạn văn:", "Bệnh viện sạch, y tá có thái độ với bệnh nhân")

if st.button("Dự đoán"):
    st.markdown("### 🔹 Kết quả dự đoán theo toàn câu:")
    full_pred = predict_sentiment(sentence)
    st.write(f"\- Dự đoán: {['Tệ', 'Trung tính', 'Tốt'][full_pred]}")

    st.markdown("### 🔸 Kết quả dự đoán theo từng mệnh đề:")
    scores, avg = score_sentence(sentence)
    if scores:
        for idx, (clause, score) in enumerate(zip(split_clauses(sentence), scores)):
            st.write(f"\- Mệnh đề {idx+1}: '{clause}' → {['Tệ', 'Trung tính', 'Tốt'][score]}")
        st.write(f"\nTrung bình: {avg:.2f} → {['Tệ', 'Trung tính', 'Tốt'][round(avg)]}")
    else:
        st.warning("Không thể tách câu thành mệnh đề hợp lệ.")

st.markdown()

# import streamlit as st
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import torch.nn.functional as F

# # Load mô hình và tokenizer
# @st.cache_resource
# def load_model():
#     model = BertForSequenceClassification.from_pretrained("sentiment_model_bert")
#     tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#     return model, tokenizer

# model, tokenizer = load_model()
# model.eval()

# # Label mapping
# id2label = {0: "Tệ", 1: "Trung tính", 2: "Tốt"}

# # Giao diện Streamlit
# st.title("Phân tích cảm xúc bình luận")

# comment = st.text_area("Nhập bình luận của bạn:", height=150)

# if st.button("Phân tích cảm xúc"):
#     if not comment.strip():
#         st.warning("Vui lòng nhập bình luận!")
#     else:
#         # Tokenize đầu vào
#         inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=128)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             probs = F.softmax(outputs.logits, dim=1)
#             pred_label = torch.argmax(probs, dim=1).item()
        
#         st.subheader("Kết quả:")
#         st.write(f"**Dự đoán:** {id2label[pred_label]}")
        
#         st.write("**Xác suất:**")
#         for i in range(3):
#             st.write(f"- {id2label[i]}: {probs[0][i].item():.4f}")

