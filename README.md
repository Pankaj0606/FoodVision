# Fruit and Vegetable Recognition System Documentation
## Project Overview
This project implements a system for recognizing fruits and vegetables from images using a Convolutional Neural Network (CNN). It consists of a training pipeline (using Jupyter Notebooks) and a web application (built with Streamlit) for real-time prediction.
**Key Features:**
*   Image classification of 36 different fruits and vegetables.
*   Training pipeline using TensorFlow/Keras.
*   Interactive web application for image upload and prediction.
**Supported Platforms/Requirements:**
*   Python 3.6+
*   TensorFlow 2.x
*   Streamlit
*   Libraries listed in `requirement.txt`
## Getting Started
### Installation and Setup
1.  **Clone the repository:**  
    Since the provided codebase is a merged file, you'll need to create the directory structure and files manually. Create a directory named `Fruit_veg_webapp` and place `labels.txt` and `main.py` inside. Also, create `LICENSE` and `requirement.txt` in the root directory. The Jupyter Notebooks and `training_hist.json` should also be placed in the root directory.
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    3.  **Install dependencies:**
    ```bash
    pip install -r requirement.txt
    
### Dependencies
The project relies on the following Python libraries:
*   tensorflow==2.10.0
*   scikit-learn==1.3.0
*   numpy==1.24.3
*   matplotlib==3.7.2
*   seaborn==0.13.0
*   pandas==2.1.0
*   streamlit
*   librosa==0.10.1
These dependencies are listed in the `requirement.txt` file and can be installed using `pip`.
## Code Structure
The project is organized as follows:
*   `Fruit_veg_webapp/`: Contains the Streamlit web application.
    *   `labels.txt`: List of recognized fruit and vegetable labels.
    *   `main.py`: Streamlit application code.
*   `LICENSE`: Contains the MIT License.
*   `requirement.txt`: Lists the Python dependencies.
*   `Testing_fruit_veg_recognition.ipynb`: Jupyter Notebook for testing the trained model.
*   `Training_fruit_vegetable.ipynb`: Jupyter Notebook for training the CNN model.
*   `training_hist.json`: JSON file containing the training history (loss and accuracy).
## API Documentation
The Streamlit application provides a user interface for image prediction. It doesn't expose a traditional API. However, the core prediction logic is encapsulated in the `model_prediction` function within `Fruit_veg_webapp/main.py`.
**Function: `model_prediction(test_image)`**
*   **Purpose:** Predicts the class of a fruit or vegetable in a given image.
*   **Input:**
    *   `test_image`: Path to the image file to be classified.
*   **Output:**
    *   `result_index`: Integer representing the index of the predicted class in the `labels.txt` file.
**Code Snippet:**
```python
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element
```
## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
