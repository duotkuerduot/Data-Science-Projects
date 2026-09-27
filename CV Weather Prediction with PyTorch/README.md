# 🌤️ Weather Image Classification using PyTorch

This repository contains a complete solution for classifying weather conditions from images using a fine-tuned EfficientNetB3 model in **PyTorch**. The task was to predict weather types from the **Mendeley Weather Dataset**, aiming to build a high-performing image classifier suitable for real-time applications like weather prediction 

---

##  Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Outputs](#outputs)
- [weather_app/ Streamlit Deployment](#weather_app-streamlit-deployment)
- [How to Use](#how-to-use)
- [Results](#results)
- [License](#license)
- [Author](#author)

---

##  Objective

The goal is to classify images into one of the following weather categories:
- **Cloudy**
- **Rain**
- **Shine**
- **Sunrise**

I built a custom PyTorch training pipeline from scratch with a focus on:
- Performance optimization
- Avoiding overfitting
- Streamlined dataset preprocessing
- Clean, reproducible outputs

---

##  Dataset

- **Source**: Mendeley Weather Dataset  
- **Classes**: 4  
- **Format**: JPG/PNG images structured in subfolders by class label  
- **Train/Test Split**: 80/20 with stratification  

---

##  Methodology

I adopted a rigorous, competition-style approach:

1. **Data Loading & Preprocessing**
   - Images resized to 300×300
   - Normalization using ImageNet mean and std
   - Augmentations: RandomHorizontalFlip, RandomRotation

2. **Model Architecture**
   - Base: `EfficientNetB3` from `torchvision.models`
   - Modified classifier head: `Linear(num_features, 4)`
   - Transfer learning: Fine-tuned only the head in Phase 1, unfroze all layers in Phase 2

3. **Training Strategy**
   - Two-phase training:
     - Phase 1: Train classifier head (frozen base)
     - Phase 2: Unfreeze and fine-tune entire model
   - Optimizer: AdamW
   - Scheduler: ReduceLROnPlateau
   - Loss: CrossEntropyLoss
   - Early stopping: patience-based

4. **Validation & Logging**
   - Accuracy and loss tracked per epoch
   - Visualization via `matplotlib` (optional)

5. **Prediction Pipeline**
   - Final test predictions saved to `pytorchsubmission.csv` for Kaggle
   - Final model saved to `model.pth` for app deployment

---

##  Model Training

The training is encapsulated in the provided Jupyter notebook:

📄 [`pytorchindax24.ipynb`](./pytorchindax24.ipynb)

After running the notebook:
- `model.pth`: Serialized PyTorch model for inference
- `pytorchsubmission.csv`: Submission-ready CSV file

These files are essential for the Streamlit app deployment (`weather_app/`).

---

##  Outputs

After successful execution of the notebook, you will have:

- `model.pth`  
  🔹Renamed from best_weather_model.pth
  🔹The fine-tuned EfficientNetB3 PyTorch model for inference.

- `pytorchsubmission.csv`  
  🔹 A CSV file formatted for Kaggle submission:
    ```csv
    id,label
    1,cloudy
    2,rain
    ...
    ```

---

##  `weather_app/` Streamlit Deployment

You can deploy your trained model into a lightweight web application with the Streamlit interface.

###  Project Structure
```
weather_app/
├── app.py                # Streamlit web app interface
├── model.pth             # Pretrained model used for predictions
├── weather_labels.json   # Label mapping for classes
```

---

###  Setup & Run

```bash
pip install -r requirements.txt
cd weather_app/
streamlit run app.py
```

---

###  Features

- Upload weather images  
- Real-time predictions  
- Displays weather condition (e.g., "Cloudy", "Rain")

---

##  How to Use

###  Option 1: Train from Scratch

1. Open `pytorchindax24.ipynb`  
2. Run all cells to train the model  
3. Generated files:
   - `model.pth`
   - `pytorchsubmission.csv`

###  Option 2: Use Pretrained Files

1. Download `model.pth` and place it in `weather_app/`  
2. Run the app as described above  

---

##  Results

| Metric         | Value            |
|----------------|------------------|
| Train Accuracy | ~100% (Phase 2)  |
| Val Accuracy   | ~95–97%          |
| Test Accuracy  | ~93–96%          |
| Inference Time | < 0.2 sec/image  |

> If train accuracy is 100% but test accuracy is significantly lower, monitor for overfitting.

###  Overfitting Mitigation Strategies

- Stronger regularization  
- More aggressive data augmentation  
- Early stopping  

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## Author

**Duot Kuer**

- Kaggle: [kaggle.com/duotkuer](https://kaggle.com/duotkuer)  
- GitHub: [github.com/duotkuer](https://github.com/duotkuer)  
- LinkedIn: [linkedin.com/in/duotkuer](https://linkedin.com/in/duotkuer)
