import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """
    import mlflow
    # Set the tracking URI to the server
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    # Set an experiment name, unique and case-sensitive
    # It will create a new experiment if the experiment with given doesn't exist
    exp = mlflow.set_experiment(experiment_name = "Bike-Rental-Pred")

    # Start RUN
    mlflow.start_run(experiment_id= exp.experiment_id)        # experiment id under which to create the current run
    
    # Log parameters
    mlflow.log_param("n_estimators", config.model_config.n_estimators)
    mlflow.log_param("max_depth", config.model_config.max_depth)
    mlflow.log_param("random_state", config.model_config.random_state)
    
    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config.features],     # predictors
        data[config.model_config.target],       # target
        test_size = config.model_config.test_size,
        random_state=config.model_config.random_state,   # set the random seed here for reproducibility
    )

    # Pipeline fitting
    bikeshare_pipe.fit(X_train, y_train)
    #y_pred = bikeshare_pipe.predict(X_test)

    # Calculate the score/error
    #print("R2 score:", r2_score(y_test, y_pred).round(2))
    #print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = bikeshare_pipe)
    # printing the score
    y_pred = bikeshare_pipe.predict(X_test)
    
    testing_r2score = r2_score(y_test, y_pred).round(2)
    print("Test R2 Score :", testing_r2score)
    
    
    # Log performance metrics
    mlflow.log_metric("training_r2score", r2_score(bikeshare_pipe.predict(X_train), y_train))
    mlflow.log_metric("testing_r2score", testing_r2score)
    
    # Load current 'production' model via 'models'
    import mlflow.pyfunc
    model_name = config.app_config.registered_model_name         #"sklearn-titanic-rf-model"
    client = mlflow.tracking.MlflowClient()

    try:
        prod_model_info = client.get_model_version_by_alias(name=model_name, alias="production")         # fetch prod-model info
        prod_model_version = int(prod_model_info.version)              # prod-model version
        new_version = prod_model_version + 1                      # new model version
        # Capture the test-accuracy-score of the existing prod-model
        prod_model_run_id = prod_model_info.run_id                   # run_id of the run associated with prod-model
        prod_run = client.get_run(run_id=prod_model_run_id)          # get run info using run_id
        prod_accuracy = prod_run.data.metrics['testing_r2score']    # get metrics values
    except Exception as e:
        print(e)
        new_version = 1


    # Log trained model
    if new_version > 1:
        if prod_accuracy < testing_r2score:
            print("Trained model is better than the existing model in production, registering its new version!")
            register_model = True
        else:
            print("Trained model is not better than the existing model in production!")
            register_model = False

    else:
        print("No existing model in production, registering a new model!")
        register_model = True


    if register_model:
        # Register new model/version of model
        mlflow.sklearn.log_model(sk_model = bikeshare_pipe, 
                                artifact_path="trained_model",
                                registered_model_name=model_name,
                                )
        # Set new model version alias to 'production'
        client.set_registered_model_alias(name=model_name, alias="production", version=str(new_version))
        
    else:
        # Do not register new version of model but log it as an artifact in this run
        mlflow.sklearn.log_model(sk_model = bikeshare_pipe, 
                                artifact_path= "trained_model")
    
    # End an active MLflow run
    mlflow.end_run()
    

if __name__ == "__main__":
    print("Re-training:", os.environ['RE_TRAIN'])
    if os.environ['RE_TRAIN']:    
       run_training()