import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from DATAREADERS import DataExtractor
import missingno as mso
import matplotlib.pyplot as plt
from DATACLEANERS import PandasMethods, UnivariateImputers, OutliersTreatment  # Import OutliersTreatment
from FEATURE_SELECTION import *
from ENCODERS import *
from CHANGERS import *
from REGRESSION import *
from clustering import clusters
from CLASSIFICATION import Classification
from model_download import *

session_models=["Bagging Classifier","Extra Tree Classifier","Decision Trees","Ada Boost Classifier","Hist Gradient Boosting Classifier","Random Forest Classifier",
                "Stacking Classifier","Voting Classifier","LinearSVC","NuSVC",
               "OneClassSVM","KNN","RadiusNeighbors","BernoulliNB","CategoricalNB","ComplementNB","GaussianNB","MultinomialNB"]
for i in session_models:
    if i not in st.session_state:
        st.session_state[i]=None
session_variables=['regressionLinear','regressionRidge','regressionRidgeCV',
                   'regressionSGD','regressionElasticNet','regressionElasticNetCV','regressionLars','regressionLarsCV',
                   'regressionLasso','regressionLassoCV',"regressionLassoLars",'regressionLassoLarsCV','regressionLassoLarsIC','regressionOMP',
                  'regressionOMPCV','regressionARD','regressionBayesianRidge','regressionMTElasticNet','regressionMTLasso','regressionMTElasticNetCV',
                   'regressionHuber','regressionQuantile','regressionRANSAC','regressionPoisson','regressionTheilSen','regressionTweedie']
for i in session_variables:
    if i not in st.session_state:
        st.session_state[i]=None


# Initialize DataExtractor object
dataextractor = DataExtractor()


# Initialize session variables if not already present
for key in ["availableDatasets", "selected_dataset"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# Title of the app
st.title("Data Science Workflow Application")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Data Upload", "Data Preprocessing", "Exploratory Data Analysis",
            "Feature Engineering","Regression", "Classification","Clustering", "Model Download","Deploy Model"
        ],
        icons=[
            "cloud-upload", "eraser", "star", "plus-circle", "code",
            "arrow-down-up", "bar-chart", "scissors", "gear"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Function to handle file uploads
def upload_file(data_type, upload_func, file_type=None):
    file = st.file_uploader(f"Upload a {data_type} file", type=file_type, key=data_type)
    if file:
        data = upload_func(file)
        if data is not None:
            dataset_name = f"STAGE-1 : DATA UPLOAD {data_type} Dataset"
            st.session_state.availableDatasets[dataset_name] = data
            st.success(f"{data_type} file uploaded successfully!")
            st.dataframe(data.head())
        
# Data Upload Section
if selected == "Data Upload":
    col1, col2 = st.columns([1, 2])

    with col1:
        readCsv = st.checkbox("Read Data From CSV")
        readExcel = st.checkbox("Read Data From Excel")
        readJson = st.checkbox("Read Data From JSON")
        readHTML = st.checkbox("Read Data From HTML")

    with col2:
        if readCsv:
            upload_file("CSV", dataextractor.readCsv, ["csv"])
        if readExcel:
            upload_file("Excel", dataextractor.readExcel, ["xls", "xlsx"])
        if readJson:
            st.session_state.readJson = dataextractor.readJson()
            if st.session_state.readJson is not None:
                st.session_state.availableDatasets["JSON Dataset"] = st.session_state.readJson
        if readHTML:
            st.session_state.readHTML = dataextractor.readHTML()
            if st.session_state.readHTML is not None:
                st.session_state.availableDatasets["HTML Dataset"] = st.session_state.readHTML

# Data Cleaning Section
elif selected == "Data Preprocessing":
    st.link_button("Click Here To Perform Data Preprocessing","https://huggingface.co/spaces/Rohith25Jan/no_code_tool_for_data_preprocessing")
elif selected == "Feature Engineering":
    #https://huggingface.co/spaces/Rohith25Jan/No-Code-Platform-To-Perform-Feature-Engineering
    st.link_button("Click Here To Perform Data Preprocessing","https://huggingface.co/spaces/Rohith25Jan/No-Code-Platform-To-Perform-Feature-Engineering")
elif selected=="Exploratory Data Analysis":
    st.link_button("Perform Exploratory Data Analysis","https://huggingface.co/spaces/Rohith25Jan/Web-Platform-To-Perform-Automated-Exploratory-Data-Analysis")       
elif selected=="Regression":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

        if selected_dataset_name:
            # Load the selected dataset from session state
            st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
            dataset = st.session_state.selected_dataset
            st.markdown("### Selected Dataset")
            st.dataframe(dataset)
            # Display missing value charts
            st.divider()
            st.markdown("### Missing Value Analysis")
            fig, ax = plt.subplots(figsize=(10, 5))
            mso.matrix(st.session_state.selected_dataset, ax=ax)
            st.pyplot(fig)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            mso.heatmap(st.session_state.selected_dataset, ax=ax1)
            st.pyplot(fig1)
            regression=Regression(st.session_state.selected_dataset)
            regression.display()


elif selected == "Classification":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

        if selected_dataset_name:
            # Load the selected dataset from session state
            st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
            dataset = st.session_state.selected_dataset
            object=Classification(dataset)
            object.display()
elif selected == "Clustering":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

        if selected_dataset_name:
            # Load the selected dataset from session state
            st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
            dataset = st.session_state.selected_dataset
            object=clusters(dataset)
            object.display()
elif selected == "Model Download":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

        if selected_dataset_name:
            # Load the selected dataset from session state
            st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
            dataset = st.session_state.selected_dataset
            object=DownloadModel(dataset)
            object.display()
                        
        
