# Welcome to Diabetes Prediction System

## About Diabetes
Diabetes is a chronic medical condition that affects how your body turns food into energy. It occurs when your blood glucose, also called blood sugar, is too high. Over time, having too much glucose in your blood can cause serious health problems. The most common types of diabetes are Type 1, Type 2, and gestational diabetes.

## Machine Learning Project
This diabetes prediction tool is developed using machine learning techniques to predict the likelihood of diabetes in individuals based on various health parameters. The model is trained on the Pima Indians Diabetes Dataset, which includes parameters such as:

- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age

## Model Accuracy
The machine learning model used in this project achieves an accuracy of approximately 80% on the test dataset. This means that the model is able to correctly predict diabetes 80% of the time based on the given health parameters.

## Online Predictor
Use the sidebar to navigate to the Diabetes Prediction section and input your health parameters to check the likelihood of diabetes. Please note that this tool is for informational purposes only and is not a substitute for professional medical advice.

[Diabetes Prediction Tool](https://bcu-diabetic-predict.streamlit.app/)

## How to Run the Project Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/diabetes-prediction-system.git
    cd diabetes-prediction-system
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Repository Structure

- `diabetes.csv`: The dataset used for training the model.
- `app.py`: The main file to run the Streamlit web app.
- `model.pkl`: The trained machine learning model.
- `requirements.txt`: The list of required packages.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The Pima Indians Diabetes Dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
- The [Streamlit](https://www.streamlit.io/) team for providing a great tool to create web apps for machine learning projects.

