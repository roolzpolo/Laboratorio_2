import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

def load_environment_variables():
    load_dotenv()

    DATASET = os.getenv("DATASET")
    TARGET = os.getenv("TARGET")
    MODEL = os.getenv("MODEL")
    TRIALS = int(os.getenv("TRIALS", 10))

    return DATASET, TARGET, MODEL, TRIALS

def preprocess_data(df, target_column=None):
    
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'preprocessor.pkl')

    return X_processed, y

def train_and_optimize_model(X, y, model_name, trials=10):
    models = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB()
    }

    param_grids = {
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'NaiveBayes': {}
    }

    model = models.get(model_name)

    if model is None:
        raise ValueError(f"Modelo {model_name} no está soportado.")

    search = RandomizedSearchCV(model, param_grids[model_name], n_iter=trials, cv=3, verbose=2, random_state=42)
    search.fit(X, y)
    
    logging.info(f"Mejores hiperparámetros para {model_name}: {search.best_params_}")
    logging.info(f"Mejor score para {model_name}: {search.best_score_}")
    
    return search.best_estimator_

def save_model(model, model_path):
    joblib.dump(model, model_path)
    logging.info(f"Modelo guardado en: {model_path}")

def train_model():
    DATASET, TARGET, MODEL, TRIALS = load_environment_variables()

    logging.info(f'Leyendo datos de entrenamiento {DATASET}')
    df = pd.read_parquet(DATASET)

    logging.info(f'Iniciando preprocesamiento de datos')
    logging.info(f'Variable objetivo: {TARGET}')
    logging.info(f'Guardando preprocesador')
    X, y = preprocess_data(df, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f'Entrenamiento y optimización de modelo: {MODEL}')
    logging.info(f'Cantidad de modelos a probar: {TRIALS}')
    model = train_and_optimize_model(X_train, y_train, MODEL, trials=TRIALS)

    logging.info('Guardando modelo')
    model_path = "models/trained_model.joblib"
    save_model(model, model_path)

    logging.info('Predicciones sobre conjunto de prueba')
    y_pred = model.predict(X_test)

    logging.info('Métricas de rendimiento')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f'Precisión: {accuracy:.4f}')
    logging.info(f'F1-Score: {f1:.4f}')
    logging.info(f'Matriz de Confusión: \n{cm}')

    return model