# AI-Spam-Email-Detector

A comprehensive project to classify emails as **spam** or **ham (non-spam)** using deep learning. This repository provides an end-to-end solution for detecting spam emails with high accuracy while ensuring a balanced and unbiased approach.

---

## 🚀 Key Features

- **Accurate Classification:** Achieved **97%+ validation accuracy** on unseen data.
- **Balanced Dataset:** Applied downsampling techniques to ensure fairness in classification.
- **Scalable Model:** Designed with modern neural network architecture suitable for text classification.

---

## 📚 Overview

Spam emails clutter inboxes and pose security risks. This project aims to create a robust spam detection system that:

1. **Enhances Inbox Experience:** Reduces junk emails while retaining important communications.
2. **Improves Security:** Detects harmful or malicious content effectively.
3. **Leverages AI:** Implements deep learning for efficient and scalable email filtering.

---

## 🛠️ How It Works

### 1. **Data Preparation**
- **Tokenization:** Converts text into numerical sequences for machine learning.
- **Padding/Truncation:** Standardizes input lengths for consistent processing.
- **Balancing:** Downsamples the majority class (ham) to ensure equal representation of spam and ham emails.

### 2. **Model Architecture**
- **Embedding Layer:** Transforms text tokens into dense semantic vectors.
- **Global Average Pooling:** Summarizes email content into a compact representation.
- **Dense Layers:** Processes and classifies data using ReLU and sigmoid activation functions.
- **Dropout Regularization:** Prevents overfitting and improves generalization.

### 3. **Training**
- Trained for **up to 30 epochs** with **early stopping** to avoid overfitting.
- Optimized using **binary cross-entropy loss** and tracked with accuracy metrics.

---

## 💡 Results

- **Validation Accuracy:** Exceeded **97%**, proving effective generalization.
- **Balanced Classification:** Prevented bias towards the majority class by equalizing data distribution.
- **Robust Design:** Achieved reliable results without overfitting through regularization and early stopping.

---
## 📊 Web Interface Screenshots

### 1. **Welcome Screen**
<img width="768" alt="Screenshot 2024-12-10 at 10 46 13 AM" src="https://github.com/user-attachments/assets/5eaa91ef-aeea-4728-ab0c-166e1007a045">


### 2. **Spam Detected**
<img width="768" alt="Screenshot 2024-12-10 at 10 47 03 AM" src="https://github.com/user-attachments/assets/ed89bc7f-2577-4d47-bd19-39d12c1251ab">


### 3. **Ham (Non-Spam) Detected**
<img width="768" alt="Screenshot 2024-12-10 at 10 48 15 AM" src="https://github.com/user-attachments/assets/0c820bf8-0930-42c7-898f-0c4f6bcf154b">



---

## 🔮 Future Enhancements

1. **Advanced Preprocessing:** Incorporate lemmatization, stemming, or pretrained embeddings (e.g., Word2Vec, GloVe).
2. **Architectural Improvements:** Explore RNNs, LSTMs, GRUs, or transformers for richer text understanding.
3. **Expanded Metrics:** Use precision, recall, F1-score, and AUC for a comprehensive performance evaluation.

---

## 🧑‍💻 How to Use

### 1. **Clone the Repository**
```bash
git clone https://github.com/Iskandarov1/AI-Spam-Email-Detector.git
cd AI-Spam-Email-Detector
```

### 2. **Create a Virtual Environment**
```bash
python3 -m venv flaskenv
```

### 3. **Activate the Virtual Environment**
```bash
source flaskenv/bin/activate
```

### 4. **Install Required Packages**
```bash
pip install flask tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### 5. **Run Your Flask Application**
```bash
python3 app.py
```



---

## 📊 Performance Metrics

- **Accuracy:** Achieved 97%+ on validation data.
- **Balanced Performance:** No significant bias toward any class.

---




## 🤝 Contributions

Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests to enhance the project.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ✉️ Contact

For questions or feedback, feel free to reach out via the repository issues or email at iskandarovsanjarbek1327@gmail.com
