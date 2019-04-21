#!/bin/bash


#Add lines for exploring effect of dropout for coastline ; do the same for flowers script;
#Create final script to run flowers shell, wait 10 minutes and run coastline shell
17:15
#Write document
18:00 RO

pip install --user pandas
pip install --user numpy
pip install --user scikit-learn

export PROJECT=$(gcloud config list project --format "value(core.project)")
export JOB_ID="coastline_${USER}_$(date +%Y%m%d_%H%M%S)"
export BUCKET="gs://${PROJECT}"
export GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
export DICT_FILE="gs://coastline-/dict.txt"
export MODEL_NAME=coastline
export VERSION_NAME=v1



# Now that we are set up, we can start processing some coastline images.
declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare  JOB_ID="coastline_${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://${PROJECT}"
declare -r GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
declare -r DICT_FILE=gs://tamucc_coastline/dict.txt
declare -r MODEL_NAME=coastline
declare -r VERSION_NAME=v1

echo
echo "Using job id: " $JOB_ID
set -v -e

#add command for creating a bucket
gsutil mb gs://${PROJECT_ID}

#Copy coastline files into our GCP bucket
gsutil cp -r gs://tamucc_coastline/dict.txt ${BUCKET}
gsutil cp -r gs://tamucc_coastline/dict_explanation.csv ${BUCKET}
gsutil cp -r gs://tamucc_coastline/labeled_images.csv ${BUCKET}
gsutil cp -r gs://tamucc_coastline/labels.csv ${BUCKET}

#Script used for splitting the coastline dataset into training and evaluation set
gsutil cp gs://coastline-238313/*.csv ./
gsutil cp *.py gs://coastline-238313

#RUN PYTHON SCRIPT FOR SPLITTING INTO TRAINING AND TEST SET
python coastline-dataset-train-test-split.py
gsutil cp labels_train_set.csv gs://coastline-238313
gsutil cp labels_test_set.csv gs://coastline-238313



cd cloud-samples/flowers
# Typically,
# the total worker time is higher when running on Cloud instead of your local
# machine due to increased network traffic and the use of more cost efficient
# CPU's.  Check progress here: https://console.cloud.google.com/dataflow
time python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "${BUCKET}/labels_test_set.csv" \
  --output_path "${GCS_PATH}/preproc/coastline_test" \
  --numWorkers 20 \
  --cloud


echo "join fsociety"

time python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://"${BUCKET}"/labels_train_set.csv" \
  --output_path "${GCS_PATH}/preproc/coastline_train" \
  --numWorkers 20 \
  --cloud

echo "hello friend"

# Training on CloudML is quick after preprocessing.  If you ran the above
# commands asynchronously, make sure they have completed before calling this one.
gcloud ml-engine jobs submit training "coastline_with_gpu_train" \
  --stream-logs \
  --scale-tier basic-gpu \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --runtime-version=1.13 \
  -- \
  --output_path "${GCS_PATH}/training/with-gpu" \
  --eval_data_paths "${GCS_PATH}/preproc/coastline_test" \
  --train_data_paths "${GCS_PATH}/preproc/coastline_train*"

export JOB_ID="coastline_gpu_${USER}_$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine jobs submit training "coastline_without_gpu" \
  --stream-logs \
  --scale-tier basic \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --runtime-version=1.13 \
  -- \
  --output_path "${GCS_PATH}/training/without-gpu" \
  --eval_data_paths "${GCS_PATH}/preproc/test*" \
  --train_data_paths "${GCS_PATH}/preproc/train*"


gcloud ml-engine versions delete $VERSION_NAME --model=$MODEL_NAME -q --verbosity none
gcloud ml-engine models delete $MODEL_NAME -q --verbosity none

# Tell CloudML about a new type of model coming.  Think of a "model" here as
# a namespace for deployed Tensorflow graphs.
gcloud ml-engine models create "$MODEL_NAME" \
  --regions eu-west-6-a

# Each unique Tensorflow graph--with all the information it needs to execute--
# corresponds to a "version".  Creating a version actually deploys our
# Tensorflow graph to a Cloud instance, and gets is ready to serve (predict).
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model" \
  --runtime-version=1.13

#Add TensorBoard line for coastline
tensorboard --port 8081 --logdir ${GCS_PATH}/training/with-gpu
tensorboard --port 8082 --logdir ${GCS_PATH}/training/without-gpu


# Models do not need a default version, but its a great way move your production
# service from one version to another with a single gcloud command.
gcloud ml-engine versions set-default "$VERSION_NAME" --model "$MODEL_NAME"

# Finally, download a coastline image so we can test online prediction.
gsutil cp \
  gs://tamucc_coastline/esi_images/IMG_2007_SecDE_Spr12.jpg \
  coastline.jpg #to be replaced with coastline path

# Since the image is passed via JSON, we have to encode the JPEG string first. <--replace daisy with coastline
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' coastline.jpg &> request.json

# Here we are showing off CloudML online prediction which is still in beta.
# If the first call returns an error please give it another try; likely the
# first worker is still spinning up.  After deploying our model we give the
# service a moment to catch up--only needed when you deploy a new version.
# We wait for 10 minutes here, but often see the service start up sooner.
sleep 10m
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json

# Remove the model and its version
# Make sure no error is reported if model does not exist
gcloud ml-engine versions delete $VERSION_NAME --model=$MODEL_NAME -q --verbosity none
gcloud ml-engine models delete $MODEL_NAME -q --verbosity none
