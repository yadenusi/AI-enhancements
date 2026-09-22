"""
Unit tests for lambda_orchestrator.py

Run with: pytest tests/test_lambda_orchestrator.py
Requires: pytest, moto, boto3
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock

os.environ.setdefault("ENDPOINT_NAME", "test-endpoint")
os.environ.setdefault("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-topic")
os.environ.setdefault("SSM_DOCUMENT_NAME", "test-runbook")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "lambda"))
import lambda_orchestrator as orch  # noqa: E402


@patch("lambda_orchestrator.ec2")
def test_get_monitored_instances_filters_tag(mock_ec2):
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Reservations": [{"Instances": [{"InstanceId": "i-abc123"}]}]}
    ]
    mock_ec2.get_paginator.return_value = mock_paginator

    result = orch.get_monitored_instances()

    assert result == ["i-abc123"]
    mock_ec2.get_paginator.assert_called_with("describe_instances")


@patch("lambda_orchestrator.sagemaker_runtime")
def test_invoke_model_parses_probability(mock_runtime):
    body = MagicMock()
    body.read.return_value = b"0.823"
    mock_runtime.invoke_endpoint.return_value = {"Body": body}

    probability = orch.invoke_model([1.0, 2.0, 3.0])

    assert probability == 0.823


@patch("lambda_orchestrator.ssm")
@patch("lambda_orchestrator.sns")
@patch("lambda_orchestrator.invoke_model")
@patch("lambda_orchestrator.build_feature_vector")
@patch("lambda_orchestrator.get_monitored_instances")
def test_handler_triggers_remediation_above_act_threshold(
    mock_instances, mock_features, mock_invoke, mock_sns, mock_ssm
):
    mock_instances.return_value = ["i-critical"]
    mock_features.return_value = [0] * 9
    mock_invoke.return_value = 0.9

    response = orch.handler({}, None)

    mock_sns.publish.assert_called_once()
    mock_ssm.start_automation_execution.assert_called_once()
    body = json.loads(response["body"])
    assert body[0]["instance_id"] == "i-critical"
    assert body[0]["probability"] == 0.9


@patch("lambda_orchestrator.ssm")
@patch("lambda_orchestrator.sns")
@patch("lambda_orchestrator.invoke_model")
@patch("lambda_orchestrator.build_feature_vector")
@patch("lambda_orchestrator.get_monitored_instances")
def test_handler_skips_action_below_warn_threshold(
    mock_instances, mock_features, mock_invoke, mock_sns, mock_ssm
):
    mock_instances.return_value = ["i-healthy"]
    mock_features.return_value = [0] * 9
    mock_invoke.return_value = 0.1

    orch.handler({}, None)

    mock_sns.publish.assert_not_called()
    mock_ssm.start_automation_execution.assert_not_called()
