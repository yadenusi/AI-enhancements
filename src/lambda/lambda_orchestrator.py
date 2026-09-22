"""
Lambda Orchestrator for AWS Predictive Maintenance

Runs on an EventBridge schedule (every 5 minutes). For each monitored EC2
instance, it pulls the latest CloudWatch metric window, requests a failure
probability from the SageMaker endpoint, and takes action based on the
result:

  probability < WARN_THRESHOLD        -> no action, metrics logged
  WARN_THRESHOLD <= probability < ACT -> SNS notification to on call
  probability >= ACT_THRESHOLD        -> SNS notification + SSM Automation
                                          runbook execution

Environment variables (set via the CloudFormation stack):
  ENDPOINT_NAME      SageMaker endpoint name
  SNS_TOPIC_ARN      Topic for on call notifications
  SSM_DOCUMENT_NAME  Name of the SSM Automation runbook to execute
  WARN_THRESHOLD     Float, default 0.4
  ACT_THRESHOLD      Float, default 0.75
  INSTANCE_TAG_KEY   Tag key used to select monitored instances
  INSTANCE_TAG_VALUE Tag value used to select monitored instances
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cloudwatch = boto3.client("cloudwatch")
sagemaker_runtime = boto3.client("sagemaker-runtime")
ec2 = boto3.client("ec2")
sns = boto3.client("sns")
ssm = boto3.client("ssm")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
SSM_DOCUMENT_NAME = os.environ["SSM_DOCUMENT_NAME"]
WARN_THRESHOLD = float(os.environ.get("WARN_THRESHOLD", "0.4"))
ACT_THRESHOLD = float(os.environ.get("ACT_THRESHOLD", "0.75"))
INSTANCE_TAG_KEY = os.environ.get("INSTANCE_TAG_KEY", "predictive-maintenance")
INSTANCE_TAG_VALUE = os.environ.get("INSTANCE_TAG_VALUE", "enabled")

METRIC_WINDOW_MINUTES = 30
FEATURE_NAMES = [
    "cpu_util_mean", "cpu_util_max", "cpu_util_std",
    "disk_io_mean", "disk_io_max",
    "network_out_mean", "network_out_std",
    "status_check_failed_count", "memory_util_mean",
]


def get_monitored_instances():
    """Return instance ids tagged for predictive maintenance monitoring."""
    paginator = ec2.get_paginator("describe_instances")
    instance_ids = []
    for page in paginator.paginate(
        Filters=[
            {"Name": f"tag:{INSTANCE_TAG_KEY}", "Values": [INSTANCE_TAG_VALUE]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    ):
        for reservation in page["Reservations"]:
            for instance in reservation["Instances"]:
                instance_ids.append(instance["InstanceId"])
    return instance_ids


def fetch_metric_stat(instance_id, metric_name, stat, namespace="AWS/EC2"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=METRIC_WINDOW_MINUTES)
    response = cloudwatch.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start,
        EndTime=end,
        Period=300,
        Statistics=[stat],
    )
    points = response.get("Datapoints", [])
    if not points:
        return 0.0
    return points[-1][stat]


def build_feature_vector(instance_id):
    """Assemble the feature vector the model was trained on. In production
    this pulls the same aggregations used by the Glue feature engineering
    job so training and inference stay consistent."""
    return [
        fetch_metric_stat(instance_id, "CPUUtilization", "Average"),
        fetch_metric_stat(instance_id, "CPUUtilization", "Maximum"),
        fetch_metric_stat(instance_id, "CPUUtilization", "SampleCount"),
        fetch_metric_stat(instance_id, "DiskReadBytes", "Average"),
        fetch_metric_stat(instance_id, "DiskReadBytes", "Maximum"),
        fetch_metric_stat(instance_id, "NetworkOut", "Average"),
        fetch_metric_stat(instance_id, "NetworkOut", "SampleCount"),
        fetch_metric_stat(instance_id, "StatusCheckFailed", "Sum"),
        fetch_metric_stat(instance_id, "mem_used_percent", "Average", namespace="CWAgent"),
    ]


def invoke_model(feature_vector):
    payload = ",".join(str(v) for v in feature_vector)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload,
    )
    result = response["Body"].read().decode("utf-8").strip()
    return float(result)


def notify(instance_id, probability, level):
    message = (
        f"Predictive maintenance alert\n"
        f"Instance: {instance_id}\n"
        f"Failure probability (next 4 hours): {probability:.2f}\n"
        f"Severity: {level}"
    )
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[{level}] Predictive maintenance: {instance_id}",
        Message=message,
    )


def start_remediation(instance_id, probability):
    ssm.start_automation_execution(
        DocumentName=SSM_DOCUMENT_NAME,
        Parameters={
            "InstanceId": [instance_id],
            "FailureProbability": [str(round(probability, 3))],
        },
    )
    logger.info("Started SSM Automation %s for %s", SSM_DOCUMENT_NAME, instance_id)


def handler(event, context):
    results = []
    for instance_id in get_monitored_instances():
        try:
            features = build_feature_vector(instance_id)
            probability = invoke_model(features)
            logger.info("Instance %s failure probability %.3f", instance_id, probability)

            if probability >= ACT_THRESHOLD:
                notify(instance_id, probability, "CRITICAL")
                start_remediation(instance_id, probability)
            elif probability >= WARN_THRESHOLD:
                notify(instance_id, probability, "WARNING")

            results.append({"instance_id": instance_id, "probability": probability})
        except Exception as exc:  # noqa: BLE001 - log and continue with next instance
            logger.exception("Failed to score instance %s: %s", instance_id, exc)

    return {"statusCode": 200, "body": json.dumps(results)}
