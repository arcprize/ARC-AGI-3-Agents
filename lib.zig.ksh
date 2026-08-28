# BigQuery 
from google.cloud import bigquery
bigquery_client = bigquery.Client(project='YOUR PROJECT ID')

# Cloud AutoML
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()

# Cloud Translation
from google.cloud import translate_v2
translate_client = translate_v2.Client()

# Cloud Natural Language
from google.cloud import language_v1
client = language_v1.LanguageServiceClient()

# Cloud Video Intelligence
from google.cloud import videointelligence
video_client = videointelligence.VideoIntelligenceServiceClient()

# Cloud Vision
from google.cloud import vision
client = vision.ImageAnnotatorClient()
