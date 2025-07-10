# Image-Processing-of-mapping-a-pattern-onto-curve
I use OpenCV and NumPy to load the flag and pattern images. 
# pattern_flag_mapper/README.md

## 🎓 Key Features
- Upload your **own white flag image with visible folds**
- Upload **any pattern image** (rectangle, logo, etc.)
- Pattern is warped to align with **natural folds** using Sobel gradients
- Realistic blending based on **lighting and curvature**
- Output preview and **download** option
- **Polished UI** with Streamlit

---

## ⚡ Technologies Used
- Python 3
- OpenCV
- NumPy
- Pillow (PIL)
- Streamlit

---

## 📁 Folder Structure
```
pattern_flag_mapper/
│
├── app.py                  # Streamlit app code
├── requirements.txt        # Python dependencies
├── sample_images/          # Sample pattern + flag images (optional)
└── Output.jpg              # Output saved after mapping
```
### 1. Run the app
```bash
streamlit run app.py
```

### 2. Open in browser
Visit `http://localhost:8501`

---

## 🖇️ How It Works
- Converts the white flag to grayscale
- Detects folds using **Sobel gradients**
- Warps pattern using a **displacement map**
- Uses histogram equalization + Gaussian blur to simulate realistic **alpha blending**

---

## 🔥 Sample Usage
1. Upload `Flag.png` (white flag with folds)
2. Upload `Pattern.png` (e.g., US flag or geometric design)
3. View warped + blended result
4. Download `Output.jpg`

---

## 🏆 Credits
Created by Jayasree for Assignment of Byteprolabs

---
