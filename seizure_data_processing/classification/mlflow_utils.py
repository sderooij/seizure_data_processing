"""utils file with functions for mlflow logging"""

import time

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from functools import partial
from itertools import starmap
from more_itertools import consume
from multiprocessing import Pool
from sklearn.metrics import get_scorer
import sqlalchemy
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from seizure_data_processing.post_processing.post_process import optimize_bias
from seizure_data_processing.post_processing.scoring import event_scoring
from tensorlibrary.learning.active import BaseActiveLearnClassifier
import cloudpickle


def set_global_variables(
    experiment_name, model_type, classifier_name, cross_val_type, dataset, patient=None
):
    """
    Set the global variables for the experiment.
    Args:
        experiment_name:
        model_type:
        classifier_name:
        cross_val_type:
        dataset:
        patient:

    Returns:

    """
    if dataset == "TUSZ":
        from config import TUSZ_DIR, SFTP_DATA_DIR

        FEATURES_DIR = TUSZ_DIR + "tle_patients/"
        SFTP_FEATURES_DIR = SFTP_DATA_DIR + "TUSZ/tle_patients/"
        FEATURE_FILE = FEATURES_DIR + "features_ordered.parquet"
        GROUP_FILE = FEATURES_DIR + "val_groups.parquet"
    elif dataset == "seize_it":
        from config import SEIZE_IT_DIR, SFTP_DATA_DIR

        if model_type == "PS" or model_type == "LOSI":
            FEATURES_DIR = SEIZE_IT_DIR + patient + "/"
        else:
            FEATURES_DIR = SEIZE_IT_DIR

        SFTP_FEATURES_DIR = SFTP_DATA_DIR + "seize_it/"
        FEATURE_FILE = FEATURES_DIR + "features_preprocessed.parquet"
        GROUP_FILE = FEATURE_FILE
    else:
        raise ValueError("Dataset not recognized")

    PARAMETER_FILE = f"parameters/parameters_{model_type}_{classifier_name}.json"

    return FEATURES_DIR, SFTP_FEATURES_DIR, FEATURE_FILE, GROUP_FILE, PARAMETER_FILE


def set_up_experiment(tracking_url, experiment_name):
    """
    Set up the experiment in mlflow. Docstring in google format.

    Args:
        tracking_url: the url of the mlflow tracking server
        experiment_name: the name of the experiment
    Returns:
        experiment_id: the id of the experiment
    """
    mlflow.set_tracking_uri(uri=tracking_url)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    return experiment_id


def get_classifier(classifier_name):
    """
    Get the classifier from the classifier name.

    Args:
        classifier_name: the name of the classifier. Options are "SVM" or "CPKRR".
    Returns:
        clf: the classifier
    """
    if classifier_name == "SVM":
        from sklearn.svm import SVC

        clf = SVC(class_weight="balanced")
    elif classifier_name == "CPKRR":
        from tensorlibrary.learning.t_krr import CPKRR

        clf = CPKRR()
    else:
        raise ValueError("Classifier name not recognized")
    return clf


def get_scaler(scaler_name):
    """
    Get the scaler from the scaler name.

    Args:
        scaler_name: the name of the scaler. Options are "min-max", "standard" or "none".
    Returns:
        scaler: the scaler
    """
    if scaler_name == "min-max":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    elif scaler_name == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
    elif scaler_name == "none":
        return None
    else:
        raise ValueError("Scaler not recognized")
    return scaler


def get_sampler(sampler_name):
    """
    Get the sampler from the sampler name.

    Args:
        sampler_name: the name of the sampler. Options are "random", "smote" or "none".
    Returns:
        sampler: the sampler
    """
    if sampler_name == "random":
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler()
    elif sampler_name == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE()
    elif sampler_name == "none":
        return None
    else:
        raise ValueError("Sampler not recognized")
    return sampler


def get_grid_search(
    model, grid_search_type, grid_search_scoring, parameters, n_jobs=-1
):
    """
    Get the grid search from the model, grid search type, grid search scoring, parameters and number of jobs.

    Args:
        model: the classifier model (must be a sklearn type model or pipeline)
        grid_search_type: the type of grid search, either "full", "random" or "none"
        grid_search_scoring: scoring metric for the grid search (e.g. "roc_auc", "accuracy", "f1", "precision", "recall", "average_precision")
        parameters: the parameters of the grid search in a dictionary.
        n_jobs: the number of jobs to run in parallel, default=-1 (all processors).
    Returns:
        grid_search: the grid search object
    """

    if len(parameters) == 0:
        grid_search = model
    else:
        hyperparams = {f"clf__{key}": parameters[key] for key in parameters.keys()}
        if grid_search_type == "full":
            from sklearn.model_selection import GridSearchCV

            grid_search = GridSearchCV(
                model, hyperparams, cv=5, scoring=grid_search_scoring, n_jobs=n_jobs
            )
        elif grid_search_type == "random":
            from sklearn.model_selection import RandomizedSearchCV

            grid_search = RandomizedSearchCV(
                model, hyperparams, cv=5, scoring=grid_search_scoring, n_jobs=n_jobs
            )
        elif grid_search_type == "none" or grid_search_type is None:
            hyperparams = {key: hyperparams[key][0] for key in hyperparams.keys()}
            grid_search = model.set_params(**hyperparams)
        else:
            raise ValueError("Grid search type not recognized")
    return grid_search


def create_pipeline(
    scaler_name,
    classifier_name,
    grid_search_type,
    grid_search_scoring,
    parameters,
    n_jobs=-1,
    under_sampling=False,
    sampling_method="random",
    sampling_ratio=0.5,
):
    """
    Create a pipeline with a scaler, classifier and grid search object.

    Args:
        scaler_name: the name of the scaler
        classifier_name: the name of the classifier
        grid_search_type: the type of grid search, either "full", "random" or "none"
        grid_search_scoring: scoring metric for the grid search (e.g. "roc_auc", "accuracy", "f1", "precision", "recall", "average_precision")
        parameters: the parameters of the grid search in a dictionary.
        n_jobs: the number of jobs to run in parallel, default=-1 (all processors).
        under_sampling: whether to use under sampling, default=False
        sampling_method: the method of under sampling, default='random'
        sampling_ratio: the ratio of under sampling = N_minority / N_majority, default=0.5

    Returns:
        grid_search: the grid search object
    """

    clf = get_classifier(classifier_name)
    if scaler_name != "none" and scaler_name is not None:
        scaler = get_scaler(scaler_name)
    else:
        scaler = None
    if not under_sampling:
        if scaler is not None:
            pipe = Pipeline(steps=[("scaler", scaler), ("clf", clf)])
        else:
            pipe = Pipeline(steps=[("clf", clf)])
    else:
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_ratio, replacement=False
        )
        if scaler is not None:
            pipe = ImbPipeline(
                steps=[("scaler", scaler), ("sampler", sampler), ("clf", clf)]
            )
        else:
            pipe = ImbPipeline(steps=[("sampler", sampler), ("clf", clf)])

    grid_search = get_grid_search(
        pipe, grid_search_type, grid_search_scoring, parameters, n_jobs=n_jobs
    )
    return grid_search


def generate_run_name(
    classifier_name, model_type, cross_val_type, patient=None, child_run=False
):
    """
    Generate a run name from the classifier name, model type, cross validation type and patient. Run name ends with a timestamp.

    Args:
        classifier_name: the name of the classifier
        model_type: the type of model, either "PS", "LOSI" or "PI"
        cross_val_type: the type of cross validation, either "LOO" or "KFold"
        patient: the patient id, only for model_type="PS"
    Returns:
        run_name: the name of the run, ends with unix timestamp.
    """
    if model_type == "PS":
        run_name = f"{classifier_name}_{model_type}_{cross_val_type}_{patient}_{int(time.time())}"
    elif model_type == "PI" or (model_type == "LOSI" and cross_val_type=='AL'):
        run_name = f"{classifier_name}_{model_type}_{cross_val_type}_{int(time.time())}"
    elif model_type == "PF" and cross_val_type=='LOSI':
        t = int(time.time())
        # round to the nearest hour
        t = t - (t %  3600)
        t = t+3600
        run_name = f"{classifier_name}_{model_type}_{cross_val_type}_{patient}_{t}"
    else:
        raise ValueError("Model type not recognized")
    return run_name


def log_group_run(
    estimator,
    group_id,
    model_type,
    classifier_name,
    patient,
    tags,
    scores,
    groups,
    signature,
    *,
    extra_table=None
):
    """
    Function to log group runs as child runs.
    Args:
        estimator: The estimator object
        group_id: The unique group id (e.g. patient id)
        model_type: The type of model, either "PS", "LOSI" or "PI"
        classifier_name: The name of the classifier
        patient: The patient id, only for model_type="PS"
        tags: The tags used in the model
        scores: The scores of the model
        groups: The groups used in the model
    Returns:
        None

    """
    if model_type == "PS" or model_type == "PF":
        run_name = f"{classifier_name}_{model_type}_{patient}_{group_id}"
    else:
        run_name = f"{classifier_name}_{model_type}_{group_id}"

    # idx = np.where(groups == group_id)[0]

    # labels = labels[idx]
    with mlflow.start_run(nested=True, run_name=run_name) as child_run:
        # Log the tags
        tags = tags.copy()
        tags["group"] = group_id  # TODO: check that these correspond to the estimator
        if model_type == "PI":
            tags["patient"] = group_id
        else:
            tags["patient"] = patient
        mlflow.set_tags(tags)
        if isinstance(groups, dict):
            mlflow.log_params(groups)

        # # get signature
        # signature = infer_signature(ann_df, output_labels)
        # Log the parameters
        if hasattr(estimator, "best_params_"):
            params = estimator.best_params_
            params = {key: params[key] for key in params.keys() if "w_init" not in key}
            mlflow.log_params(params)
        else:
            params = estimator.get_params()
            if "clf" in params.keys():
                # get params with "clf__" prefix without the "w_init" parameter
                params = {key: params[key] for key in params.keys() if "clf__" in key and "w_init" not in key}
            else:
                params = {key: params[key] for key in params.keys() if "w_init" not in key}
            if isinstance(estimator, BaseActiveLearnClassifier):
                # remove params with init in key
                params = {key: params[key] for key in params.keys() if "init" not in key}
                # include "included groups"
                params["included_groups"] = estimator.included_groups
                params["model_params"] = {key: estimator.model_params[key] for key in estimator.model_params.keys() if "w_init" not in key}

            mlflow.log_params(params)
        if extra_table is not None:
            mlflow.log_table(extra_table, artifact_file="parameters.json")
        # Log the scores #TODO: write a function to log the scores
        mlflow.log_metrics(scores)
        # Log the model
        if hasattr(estimator, "best_estimator_"):
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator.best_estimator_,
                artifact_path="sk_model",
                signature=signature,
            )
        else:
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator,
                artifact_path="sk_model",
                signature=signature
            )

    return None


def log_group_runs(
    val_dict,
    model_type,
    classifier_name,
    patient,
    tags,
    scores,
    groups,
    signature,
):
    """
    Function to log group runs as child runs.
    Args:
        val_dict: A dictionary containing the validation results
        model_type: The type of model, either "PS", "LOSI" or "PI"
        classifier_name: The name of the classifier
        patient: The patient id, only for model_type="PS"
        tags: The tags used in the model
        ann_df: The ann_df used in the model
        labels: The labels used in the model
        unique_groups: The unique groups used in the model
    Returns:
        None

    """
    results = []
    unique_groups = np.unique(groups)
    for i, estimator in enumerate(val_dict["estimator"]):
        group_id = unique_groups[i]
        results.append(
            log_group_run(
                estimator,
                group_id,
                model_type,
                classifier_name,
                patient,
                tags,
                scores[group_id],
                groups,
                signature,
            )
        )

    return results


def log_parent_run(
    experiment_id,
    run_name,
    grid_search,
    val_dict,
    groups,
    feature_file,
    group_file,
    tags,
    model_type,
    classifier_name,
    predictions,
    scores,
    signature,
    child_runs=True,
    patient=None,
    temp_dir="temp/",
    crossval_type="LOPO",
    hyperparams=None,
    save_cv_obj=False,
):

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Log the tags
        mlflow.set_tags(tags)
        # Log the parameters
        grid_params = grid_search.get_params()
        grid_params = {key: grid_params[key] for key in grid_params.keys() if "estimator" not in key}
        grid_params = {key: grid_params[key] for key in grid_params.keys() if "w_init" not in key}# to prevent
        if hyperparams is not None:
            if crossval_type == "LOSI":
                grid_params.update(hyperparams)
        # too much data in the parameters
        mlflow.log_params(grid_params)
        mlflow.log_param("feature_file", feature_file)
        mlflow.log_param("group_file", group_file)
        if save_cv_obj:
            with open(f"{temp_dir}/cv_obj.pkl", "wb") as f:
                cloudpickle.dump(val_dict, f)
            mlflow.log_artifact(f"{temp_dir}/cv_obj.pkl")

        # log the predictions to a parquet file
        predictions.to_parquet(f"{temp_dir}/output.parquet")
        mlflow.log_artifact(f"{temp_dir}/output.parquet")
        # Log the scores
        # log mean and std of scores over the groups
        for key, value in val_dict.items():
            if isinstance(value, np.ndarray) and not isinstance(value[0], str):
                mlflow.log_metric(f"{key}_mean", np.mean(value))
                mlflow.log_metric(f"{key}_std", np.std(value))
            else:
                continue

        # for key in val_dict.keys():
        #     if key not in ["estimator", "experiment_id", "indices"]:
        #         if isinstance(val_dict[key], np.ndarray):
        #             mlflow.log_metric(f"{key}_mean", np.mean(val_dict[key]))
        #             mlflow.log_metric(f"{key}_std", np.std(val_dict[key]))
        #         else:
        #             continue
        # TODO add ROC and PR curves
        # save the models, outputs and performance of the groups
        # unique_groups = np.unique(groups)

        # log group runs

        if child_runs:
            # log_child_run = partial(
            #     log_group_run,
            #     model_type=model_type,
            #     classifier_name=classifier_name,
            #     patient=patient,
            #     tags=tags,
            #     ann_df=ann_df,
            #     labels=labels,
            #     metrics=metrics,
            #     groups=groups,
            # )
            # with Pool() as pool:
            #     predictions = pool.starmap(log_child_run, zip(val_dict["estimator"], unique_groups))
            if crossval_type == "LOPO" or crossval_type == "LOSO":
                log_group_runs(
                    val_dict,
                    model_type,
                    classifier_name,
                    patient,
                    tags,
                    scores,
                    groups,
                    signature,
                )
            elif crossval_type == "LOSI":
                for i, estimator in enumerate(val_dict["estimator"]):
                    group_id = i
                    # mlflow.log_param('train_group', val_dict["groups"][i]['train'])
                    # mlflow.log_param('test_group', val_dict["groups"][i]['test'])
                    log_group_run(
                        estimator,
                        group_id,
                        model_type,
                        classifier_name,
                        patient,
                        tags,
                        scores[group_id],
                        val_dict["groups"][i],
                        signature,
                    )
            # predictions = np.concatenate(predictions)
            # scores = event_scoring(
            #     np.sign(predictions[:, 0]),
            #     predictions[:, 1],
            #     overlap=0.5,
            #     seglen=2,
            #     arp=10.0,
            #     min_duration=10.0,
            #     pos_percent=0.8,
            # )

        # mlflow.log_metrics(scores["overall"])

    return None
