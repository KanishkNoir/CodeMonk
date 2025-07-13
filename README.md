# Fashion Product Classifier

A deep learning application that predicts 4 different attributes from a single fashion product image using a ResNet-18 based multi-head neural network.

## 🌐 Live Demo

**Try it now**: [https://fashionclass.streamlit.app/](https://fashionclass.streamlit.app/)

No installation required! Just upload an image or paste an image URL to get instant predictions.

**Kaggle Notebook**: [https://www.kaggle.com/code/kanishknoir/codemonk-kanishk-pratap-singh](https://www.kaggle.com/code/kanishknoir/codemonk-kanishk-pratap-singh)

## 🚀 Quick Start

### Option 1: Use Live Demo (Recommended)
Simply visit [https://fashionclass.streamlit.app/](https://fashionclass.streamlit.app/) - no setup required!

### Option 2: Run Locally

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CodeMonk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run product_prediction/frontend/app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Features

### What the Model Predicts
- **Color**: 46 different colors (Red, Blue, Black, etc.)
- **Product Type**: 143 different types (T-shirt, Jeans, Shoes, etc.)
- **Season**: 4 seasons (Spring, Summer, Fall, Winter)
- **Gender**: 5 categories (Men, Women, Boys, Girls, Unisex)

### How to Use
1. **Upload an image** or **paste an image URL**
2. **Click classify** to get predictions
3. **View results** for all 4 attributes instantly

## 📁 Project Structure

```
CodeMonk/
├── product_prediction/
│   ├── frontend/
│   │   └── app.py              # Streamlit web application
│   └── model/
│       ├── model.py            # ResNet-18 multi-head architecture
│       ├── codemonk_model.pth  # Trained model weights (43MB)
│       ├── class_mappings.json # Label mappings for all categories
│       ├── transforms.pkl      # Image preprocessing pipeline
│       └── __init__.py
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: ResNet-18 (11M parameters)
- **Multi-Head Design**: Shared backbone with 4 task-specific heads
- **Input**: 224x224 RGB images
- **Output**: 4 separate predictions with confidence scores

### Dependencies
- `streamlit`: Web interface
- `torch`: Deep learning framework
- `torchvision`: Computer vision utilities
- `Pillow`: Image processing
- `requests`: HTTP requests for URL images

## 🎯 Why We Chose ResNet-18

### Problem Statement
We need to predict 4 different attributes from a single fashion product image:
- Color (46 different colors)
- Product Type (143 different types like T-shirt, shoes, etc.)
- Season (4 seasons)
- Gender (5 categories)

### 1. **Perfect Size for Our Dataset**
- Our dataset has 44,424 images - ResNet-18 is big enough to learn patterns but not so big that it overfits
- Has 11 million parameters which is just right for fashion image classification
- Faster to train than bigger models like ResNet-50 or ResNet-101

### 2. **Multi-Head Architecture - One Model, Four Predictions**
- **Shared Backbone**: ResNet-18 learns common visual features (edges, textures, patterns) that all tasks need
- **Task-Specific Heads**: 4 separate final layers, each specialized for one prediction:
  - Color Head: Outputs 46 color probabilities
  - Article Head: Outputs 143 product type probabilities  
  - Season Head: Outputs 4 season probabilities
  - Gender Head: Outputs 5 gender probabilities
- **Why This Works**: Lower layers learn "what does this fabric look like?" while upper heads learn "what color/type/season/gender is this?"
- **Efficiency**: One model does 4 jobs instead of training 4 separate models

### 3. **Works Well with Our Data Quality**
- Our images have high brightness (84.2%) and very low blur (0.5%)
- This means ResNet-18's feature extraction layers will work effectively
- Good image quality = better feature learning = better predictions

### 4. **Handles Our Data Challenges**
- **Class Imbalance**: Our dataset has more men's products and summer items - ResNet-18 can handle this
- **Difficulty Levels**: Some predictions are harder (article type) than others (gender) - ResNet-18 adapts well
- **Correlations**: Our analysis showed season and article type are connected - shared features help both predictions

### 5. **Multi-Head Learning Advantages**
- **Shared Learning**: All tasks help each other learn better visual features
- **Correlation Benefits**: Our EDA showed season and article type are strongly connected (0.685 correlation) - the model learns this naturally

### 6. **Proven Track Record**
- ResNet-18 is widely used for image classification tasks
- Has pre-trained weights from ImageNet which gives us a head start
- Well-documented and reliable architecture
- Multi-head approach is standard practice for multi-task learning

## 📊 Model Performance

### Baseline Accuracy (Most Common Class)
| Attribute | Most Common Class | Baseline Accuracy |
|-----------|-------------------|-------------------|
| Gender | Men | 49.85% |
| Article Type | Tshirts | 15.91% |
| Base Colour | Black | 21.91% |
| Season | Summer | 48.38% |

### Training Results After 10 Epochs
| Attribute | Train Accuracy | Validation Accuracy | Improvement vs Baseline |
|-----------|---------------|---------------------|------------------------|
| Gender | 90.17% | 88.74% | +38.89% |
| Article Type | 81.56% | 78.80% | +62.89% |
| Base Colour | 55.39% | 51.94% | +30.03% |
| Season | 73.35% | 70.05% | +21.67% |

**Training Loss**: 2.2264 | **Validation Loss**: 4.0071

### Key Insights
- **Article Type** shows the most significant improvement (62.89% over baseline)
- **Gender** achieves the highest absolute accuracy (88.74% validation)
- **Base Colour** is the most challenging task but still shows 30% improvement
- All attributes significantly outperform baseline accuracy

### Expected Performance
- **ResNet-50**: Too heavy for our dataset size, would train slower
- **Simple CNN**: Not powerful enough for 143 different product types
- **EfficientNet**: Good option but ResNet-18 is simpler and more reliable
- **4 Separate Models**: Would require 4x more training time and resources

### Model Files
- **codemonk_model.pth**: Pre-trained model weights (43MB)
- **class_mappings.json**: Maps class indices to human-readable labels
- **transforms.pkl**: Image preprocessing pipeline used during training

## 💡 Usage Examples

### Supported Image Formats
- **File Upload**: JPG, JPEG, PNG
- **URL Input**: Any publicly accessible image URL
- **Image Size**: Automatically resized to 224x224 pixels

### Sample Workflow
1. Start the app: `streamlit run product_prediction/frontend/app.py`
2. Choose "Upload Image" or "Image URL"
3. Select/paste your fashion image
4. Get instant predictions for all 4 attributes

## 🛠️ Development

### Running in Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run product_prediction/frontend/app.py --server.fileWatcherType poll
```

### Model Architecture Code
The multi-head ResNet-18 implementation is in `product_prediction/model/model.py`:
- Shared ResNet-18 backbone (without final FC layer)
- 4 separate linear heads for each prediction task
- Forward pass returns all 4 predictions simultaneously
