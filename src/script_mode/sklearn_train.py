import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# inference functions ---------------
# Importantísima esta función, debe estar presente para el hosting del modelo
# sin esta función, levantar el endpoint como un built in es imposible

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def predict_fn(input_data, model):
    """A default predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn
    Returns: a prediction
    """
    
    output = model.predict(input_data)
    return output

# Obligatorio mantener main guard si utilizamos el script para el hosting del modelo 
# por ahora deberíamos mantener el training y el hosting en el mismo script, no encontré 
# la manera de poner los scripts por separado

if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    
    # Acá se establecen los hyperparameters que queremos declarar para tunear.
    # Por ejemplo, le pasamos estos dos, si en el training job le quiero agregar
    # max_depth por ejemplo, el modelo no lo va a reconocer como hyperparámetro y se va a caer el  proceso
    # Dado que nunca lo declaré como un argumento
    
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories
    # Más informacion sobre las variables de enviorement https://github.com/aws/sagemaker-containers (abajo en el README)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="iris_train_dataset.csv")
    parser.add_argument("--test-file", type=str, default="iris_test_dataset.csv")
    
    args, _ = parser.parse_known_args()

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("building training and testing datasets")
    X_train = train_df.drop("target", axis=1)
    X_test = test_df.drop("target", axis=1)
    y_train = train_df["target"]
    y_test = test_df["target"]

    # train
    #Aca instanciamos el modelo, dentro de los hyperparameters ponemos los args.
    # Lo que se hardcodea acá se pierde, los args los podemos almacenar en DynamoDB como en los built in
    print("training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1
    )

    model.fit(X_train, y_train)

    
    # Se elije la métrica con la que vamos a evaluar, importantísimo todo el block que sigue
    # print abs error
    print("validating model")
    abs_err = np.abs(model.predict(X_test) - y_test)

    
    #Dado que el container de SKLearn va a ejecutar el script, los logs del training y el HPT van a estar implícitos.
    # Una de las soluciones para poder trakear los logs a través de CloudWatch cuando se utilicen los jobs (tanto de training como HPT)
    # es hacer un print con las métricas, después con Regex hacemos la lectura de logs, va a quedar más claro abajo
    
    # print couple perf metrics
    for q in [10, 50, 90]:
        print("AE-at-" + str(q) + "th-percentile: " + str(np.percentile(a=abs_err, q=q)))

    #Persistencia del modelo, cuando nos toque levantar el endpoint, lo que hace en un built in es serializar el modelo y disponibilizarlo a traves del
    #Endpoint, simil a como lo hacíamos con nuestros modelos en el AICORE
    
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
    print(args.min_samples_leaf)