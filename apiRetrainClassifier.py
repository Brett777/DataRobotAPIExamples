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
# Windows path example # path = "C:\\Users\\Brett\\Downloads\\datarobot_examples\\examples\\time_series\\"
# Mac path example # path = '/Users/brett.olmstead/Downloads/Demo Data/'
filename = '10K_Lending_Club_Loans.csv'
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
project_name = 'LendingClub_PROJECT_1_{}'.format(now)

# Create a project
print("Starting Project. Uploading file: " + str(filename))
proj = dr.Project.create(sourcedata=filename,
                         project_name=project_name,
                         max_wait=3600)

print('Project ID: {}'.format(proj.id))


print("Setting AutoPilot to Manual Mode.")
print("Analyzing data...")
proj.set_target(
    mode=dr.AUTOPILOT_MODE.MANUAL,
    target='is_bad',
    max_wait=3600,
    worker_count=-1
)

# Get a list of blueprints that will work for this project
blueprintMenu = proj.get_blueprints()

# View the blueprints as a dataframe
blueprintDF = pd.DataFrame(blueprintMenu)

# Get a blueprint
blueprint = blueprintMenu[0]

# Train the model from the blueprint, and wait until it's done before continuing
# proj.train_datetime(blueprint.id)

print("Training model from: " + str(blueprint))
model_job = proj.train(blueprint)
new_model = wait_for_async_model_creation(
    project_id=proj.id,
    model_job_id=model_job,
    max_wait=1200
)
print("Done training.")

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
project_name2 = 'LendingClub_PROJECT_2_{}'.format(now)

print("Creating a new project called: " + str(project_name2))
print("Uploading file " + str(filename) + "...")
proj2 = dr.Project.create(sourcedata=filename,
                          project_name=project_name2,
                          max_wait=3600)


print("Setting AutoPilot Mode to Manual.")
print("Analyzing data...")
proj2.set_target(
    mode=dr.AUTOPILOT_MODE.MANUAL,
    target='is_bad',
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
print("Training model from: " + str(blueprint))
model_job = proj2.train(blueprint)
new_model = wait_for_async_model_creation(
    project_id=proj2.id,
    model_job_id=model_job,
    max_wait=1200
)
print("Done training.")

# Get the model
proj2.get_models()
my_model = proj2.get_models()[0]

# Apply the advanced tuning parameters that we got in the previous project
print("Beginning Advanced Tuning...")
tune = my_model.start_advanced_tuning_session()
for i in range(0, len(tuning_parameters)):
    task_name = tuning_parameters.task_name[i]
    parameter_id = tuning_parameters.parameter_id[i]
    value = tuning_parameters.current_value[i]
    tune.set_parameter(task_name=task_name, parameter_id=parameter_id, value=value)

job = tune.run()
new_model = job.get_result_when_complete(max_wait=1200)
new_model

print("Advanced Tuning complete.")
print("Unlocking Holdout.")
proj2.unlock_holdout()

print("Frozen run beginning.")
new_model.request_frozen_model(sample_pct=100)

print("All done.")
