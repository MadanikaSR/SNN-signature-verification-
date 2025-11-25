# ✒️ Signature Verification Using the CEDAR Dataset

A traditional **machine-learning** system to classify handwritten signatures as **Genuine** or **Forged** using the **CEDAR Signature Dataset**.
This project uses **simple, interpretable, non-deep-learning image-processing techniques**.

---

## 📁 Dataset: CEDAR Signatures

The CEDAR dataset contains **55 folders**, each containing:

* **24 genuine signatures**
* **24 forged signatures**

Total: **55 × 48 = 2640 images**

---

## 🎯 Model Used

### **Logistic Regression**

**Reasons:**

* Ideal for small datasets
* Works well with linear or moderately complex features
* Extremely fast to train
* Easy to interpret

---

## 🧠 Project Workflow

1. Load dataset folders (`1` to `55`)
2. Read all signature images
3. Extract numerical features
4. Assign labels:

   * **Original → 1**
   * **Forged → 0**
5. Split into train/test sets
6. Train logistic regression classifier
7. Evaluate the model
8. Predict on new signature images

---

## 📦 Installation

Install dependencies:

```bash
pip install numpy opencv-python scikit-learn matplotlib
```

---

## ▶️ Running the Program

Project structure:

```
your_project/
 ├── CEDAR/
 ├── signature_model.py
 ├── README.md
```

Run:

```bash
python signature_model.py
```

---

## 💾 Files Automatically Generated

| File                | Purpose                           |
| ------------------- | --------------------------------- |
| `trained_model.pkl` | Saved logistic regression model   |
| `feature_data.npy`  | Extracted features for all images |
| `label_data.npy`    | Labels for all samples            |

---

## 📊 Evaluation Metrics

After training, the script automatically prints:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

---

## 📘 Notes

* Works on **Windows**, **Linux**, and **MacOS**
* Requires **Python 3.7+**
* Designed for **education**, clarity, and simplicity
* No deep learning, no unnecessary complexity
* Easy to extend with more features or better models

---

## 🏆 Acknowledgements

* CEDAR Signature Database
* OpenCV
* NumPy
* Scikit-Learn

---

## 📜 License

This project is open for **educational and research use**.
Feel free to modify or extend it.
