import datarobot as dr
from datarobot.models.modeljob import wait_for_async_model_creation
import datetime
import pandas as pd
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Start the DataRobot python client
dr.Client()

# Set some variables
# path = "C:\\Users\\Brett\\Downloads\\datarobot_examples\\examples\\time_series\\"
path = '/Users/brett.olmstead/Downloads/examples/time_series/'
filename = 'DR_Demo_Sales_Multiseries_training.xlsx'
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
project_name = 'DR_Demo_Retail_Multiseries_PROJECT_1_{}'.format(now)

# Create a project
print("Starting Project. Uploading file: " + str(filename))
proj = dr.Project.create(sourcedata=path + filename,
                         project_name=project_name,
                         max_wait=3600)

print('Project ID: {}'.format(proj.id))

# What projects are there?
# my_projects = dr.Project.list()
# proj = my_projects[0]

print("Configuring Time Series settings.")
# Set up a time series project
time_partition = dr.DatetimePartitioningSpecification(
    use_time_series=True,
    datetime_partition_column='Date',
    autopilot_data_selection_method='duration',
    feature_derivation_window_start='-90',
    feature_derivation_window_end='0',
    forecast_window_start='1',
    forecast_window_end='28',
    multiseries_id_columns=['Store']  # in this demo dataset, series are retail stores
)

# manually confirm time step and time unit are as expected
datetime_feature = dr.Feature.get(proj.id, 'Date')
multiseries_props = datetime_feature.get_multiseries_properties(['Store'])
print(multiseries_props)

# manually check out the partitioning settings like feature derivation window and backtests
# to make sure they make sense before moving on
full_part = dr.DatetimePartitioning.generate(proj.id, time_partition)
print(full_part.feature_derivation_window_start, full_part.feature_derivation_window_end)
print(full_part.to_dataframe())

print("Setting AutoPilot to Manual Mode.")
print("Deriving time series features and analyzing data...")
proj.set_target(
    mode=dr.AUTOPILOT_MODE.MANUAL,
    target='Sales',
    partitioning_method=time_partition,
    max_wait=3600,
    worker_count=-1
)

# Get a list of blueprints that will work for this project
blueprintMenu = proj.get_blueprints()

# View the blueprints as a dataframe
blueprintDF = pd.DataFrame(blueprintMenu)

# Get a blueprint
blueprint = blueprintMenu[4]

# Train the model from the blueprint, and wait until it's done before continuing
# proj.train_datetime(blueprint.id)

print("Training model from the original project against new data.")
model_job = proj.train_datetime(blueprint.id)
new_model = wait_for_async_model_creation(
    project_id=model_job.project_id,
    model_job_id=model_job.id,
    max_wait=1200
)
print("Training Done.")

# Get the model
proj.get_models()
my_model = proj.get_models()[0]

print("Getting advanced tuning parameters.")
# Get the tuning parameters
advanced_tune_params = my_model.get_advanced_tuning_parameters()
tuning_parameters = advanced_tune_params["tuning_parameters"]
tuning_parameters = pd.DataFrame.from_dict(advanced_tune_params["tuning_parameters"])

##########################
# CREATE A NEW PROJECT    #
##########################
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
project_name2 = 'DR_Demo_Retail_Multiseries_PROJECT_2_{}'.format(now)

print("Creating a new project called: " + str(project_name2))
print("Uploading file " + str(filename) + "...")
proj2 = dr.Project.create(sourcedata=path + filename,
                          project_name=project_name2,
                          max_wait=3600)

# Set up a NEW time series project
time_partition = dr.DatetimePartitioningSpecification(
    use_time_series=True,
    datetime_partition_column='Date',
    autopilot_data_selection_method='duration',
    feature_derivation_window_start='-90',
    feature_derivation_window_end='0',
    forecast_window_start='1',
    forecast_window_end='28',
    multiseries_id_columns=['Store']  # in this demo dataset, series are retail stores
)

# manually confirm time step and time unit are as expected
datetime_feature = dr.Feature.get(proj2.id, 'Date')
multiseries_props = datetime_feature.get_multiseries_properties(['Store'])
print(multiseries_props)

# manually check out the partitioning settings like feature derivation window and backtests
# to make sure they make sense before moving on
full_part = dr.DatetimePartitioning.generate(proj2.id, time_partition)
print(full_part.feature_derivation_window_start, full_part.feature_derivation_window_end)
print(full_part.to_dataframe())

print("Setting AutoPilot Mode to Manual.")
print("Deriving time series features and analyzing data...")
proj2.set_target(
    mode=dr.AUTOPILOT_MODE.MANUAL,
    target='Sales',
    partitioning_method=time_partition,
    max_wait=3600,
    worker_count=-1
)

# Get a list of blueprints that will work for this project
# blueprintMenu = proj.get_blueprints()

# View the blueprints as a dataframe
# blueprintDF = pd.DataFrame(blueprintMenu)

# Get the 43rd blueprint
# blueprint = blueprintMenu[43]

# Train the model from the blueprint
print("Training model from the original project against new data.")
model_job = proj2.train_datetime(blueprint.id)
new_model = wait_for_async_model_creation(
    project_id=model_job.project_id,
    model_job_id=model_job.id,
    max_wait=1200
)
print("Training Done.")

# Get the model
proj2.get_models()
my_model = proj2.get_models()[0]

# Apply the advanced tuning parameters that we got in the previous project
print("Beginning Advanced Tuning...")
tune = my_model.start_advanced_tuning_session()
for i in range(0, len(tuning_parameters)):
    task_name = tuning_parameters.task_name[i]
    parameter_name = tuning_parameters.parameter_name[i]
    value = tuning_parameters.current_value[i]
    tune.set_parameter(task_name=task_name, parameter_name=parameter_name, value=value)

job = tune.run()
new_model = job.get_result_when_complete(max_wait=1200)
new_model

print("Advanced Tuning complete.")
print("Unlocking Holdout.")
proj2.unlock_holdout()

print("Frozen run beginning.")
new_model.request_frozen_datetime_model()

print("All done.")
