"""
SerializeImageData: lambda function to serialize images
"""
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    response = s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

"""
ClassificationInference: lambda for inference
"""

import json
import base64
import boto3
#from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-01-20-16-58-14-623"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                       ContentType='image/png',
                                       Body=image)

    # For this model the IdentitySerializer needs to be "image/png"
    #predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    #inferences = predictor.predict(payload)
    
    # We return the data back to the Step Function    
    event["inferences"] = json.loads(predictor['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

"""
InferenceFilter : Lambda function that filters low confidence predictions
"""
import json


THRESHOLD = .85

def lambda_handler(event, context):
    # Get the inferences from the event
    inferences = event["inferences"]
    
    # Check if any values in any inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }