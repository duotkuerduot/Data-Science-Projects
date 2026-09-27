# Mastitis Disease Detection in Cows 🐄

## 📌 Problem Statement
Mastitis is one of the most common and costly diseases in dairy farming, caused by inflammation of the udder tissue.  
It leads to:
- Reduced milk yield and quality  
- Economic losses for farmers  
- Potential animal welfare concerns  

**Early detection is crucial** to prevent the spread of infection and reduce financial losses. However, many small-scale farmers lack access to veterinary diagnostic tools.  

This project applies **computer vision with deep learning** to automatically detect mastitis from cow teat images, providing a low-cost and scalable solution for precision agriculture.  

---

## 📊 Dataset
We used the [Mastitis Disease Detection Dataset](https://www.kaggle.com/datasets/sivaprathishsiva/mastitis-disease-detection) from Kaggle.  

### Structure:
├── mastitis.ipynb # Jupyter notebook (full pipeline)
|├── README.md # Documentation
|└── mastitis_resnet18.pth # Saved trained model (after training)


---

## ✨ Acknowledgments
- Dataset: [Kaggle - Mastitis Disease Detection](https://www.kaggle.com/datasets/sivaprathishsiva/mastitis-disease-detection)  
- Pretrained model: PyTorch `torchvision.models.resnet18`  
