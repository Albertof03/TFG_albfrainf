# TFG - Earthquake Time Series Prediction and Visualization

This is the repository of my Bachelor's Degree Final Project titled:  
"Time Series Analysis for the Prediction and Management of Seismic Data using Artificial Intelligence", focused on the automatic ingestion of seismic data, analysis and prediction using Machine Learning models, and the interactive visualization of results through a web application.

## System Deployment Manual

This manual describes step-by-step how to install, configure, and run locally the complete system developed for the Degree Final Project.

### Technologies Used

- Backend: Node.js + Express  
- Frontend: Svelte + Leaflet + Vite  
- Database: PostgreSQL + PostGIS extension  
- Prediction: Python (Pandas, Numpy, Scikit-learn, Optuna, tsfresh, SHAP)  
- Others: Git, VS Code

---

## Development Environment Setup

Make sure you have the following tools installed with the recommended versions:

- PostgreSQL 17.0 + PostGIS  
- Python 3.12  
- Java 20.0.2  
- pip 25.0.1  
- Node.js 20.11.0 + npm 10.9.0  
- Visual Studio Code 1.99.3  

---

## Deployment Steps

### 1. Clone the repository

git clone https://github.com/Albertof03/albfrainf_TFG.git

### 2. Install Python dependencies

pip install -r requirements.txt

### 3. Install JavaScript dependencies

npm install --prefix TFG_ALBERTO_DATOS  
npm install --prefix TFG_ALBERTO_WEB/backend  
npm install --prefix TFG_ALBERTO_WEB/frontend/TFG

### 4. Create and configure the database

createdb -U your_user new_database  
psql -U your_user -d new_database -c "CREATE EXTENSION postgis;"  
pg_restore -U your_user -d new_database backup_basedatos.dump

### 5. Run the automation script

From VS Code or terminal:

python automatizacion.py

This script performs the following tasks:

- Automatically generates the `.env` configuration file  
- Exports data from PostgreSQL to CSV  
- Executes the predictive model (average magnitude and number of earthquakes)  
- Loads predictions into materialized views  
- Exports final predictions  
- Builds the frontend (npm run build)  
- Optionally launches the local backend server  

### 6. Access the web application

Open your browser and go to:

http://localhost:SPECIFIED_PORT

---

## Developed by:

- Author: Alberto Fraile Infantes
