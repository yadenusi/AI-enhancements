# Predictive Maintenance for AWS EC2 Fleets

Unit 6 Choose Your Own AI Enhancement Project. This implementation predicts
EC2 instance failures before they occur and triggers pre-emptive
remediation, using CloudWatch, S3, AWS Glue, Amazon SageMaker, Lambda,
EventBridge, SNS, and Systems Manager Automation.

## Repository layout

```
infrastructure/   CloudFormation stack for the supporting AWS resources
src/lambda/       Lambda orchestrator that scores instances and takes action
src/glue/         Glue ETL job that builds the training feature set
src/sagemaker/    SageMaker training script (XGBoost)
automation/       SSM Automation remediation runbook
tests/            Unit tests for the Lambda orchestrator
docs/             Architecture diagram
```

## Deployment steps

1. **Data lake and IAM resources**
   ```
   aws cloudformation deploy \
     --template-file infrastructure/cloudformation_stack.yaml \
     --stack-name predictive-maintenance \
     --capabilities CAPABILITY_NAMED_IAM \
     --parameter-overrides OnCallEmail=oncall@example.com
   ```

2. **Enable metric collection.** Attach the CloudWatch agent to monitored
   instances and tag them `predictive-maintenance=enabled`. Confirm
   `CPUUtilization`, `DiskReadBytes`, `NetworkOut`, `StatusCheckFailed`, and
   `mem_used_percent` are being published.

3. **Deliver metrics to the data lake.** Configure a CloudWatch metric
   stream (or a scheduled export) to land Parquet files under
   `s3://<bucket>/raw/cloudwatch-metrics/`.

4. **Run feature engineering.** Deploy `src/glue/feature_engineering_job.py`
   as a Glue job (Glue 4.0, Python 3) and schedule it daily via
   EventBridge. It writes curated, labeled features to
   `s3://<bucket>/curated/features/`.

5. **Train the model.** Package `src/sagemaker/train.py` into a SageMaker
   Pipeline using the built in XGBoost container, pointing the `train`
   channel at the curated features prefix. Schedule the pipeline weekly.
   Example (SageMaker Python SDK):
   ```python
   from sagemaker.xgboost import XGBoost

   estimator = XGBoost(
       entry_point="train.py",
       source_dir="src/sagemaker",
       framework_version="1.7-1",
       instance_type="ml.m5.xlarge",
       instance_count=1,
       role=sagemaker_role,
       hyperparameters={"max-depth": 5, "eta": 0.2, "num-round": 150},
   )
   estimator.fit({"train": "s3://<bucket>/curated/features/"})
   ```

6. **Deploy the endpoint.**
   ```python
   predictor = estimator.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.large",
       endpoint_name="predictive-maintenance-endpoint",
   )
   ```

7. **Package and deploy the Lambda function.** Zip
   `src/lambda/lambda_orchestrator.py` with its dependencies and update the
   `OrchestratorFunction` resource in the CloudFormation stack, or deploy
   with the AWS CLI:
   ```
   cd src/lambda && zip -r ../../orchestrator.zip . && cd ../..
   aws lambda update-function-code \
     --function-name predictive-maintenance-orchestrator \
     --zip-file fileb://orchestrator.zip
   ```

8. **Deploy the remediation runbook.**
   ```
   aws ssm create-document \
     --name predictive-maintenance-remediation \
     --document-type Automation \
     --document-format YAML \
     --content file://automation/ssm_remediation_runbook.yaml
   ```

9. **Run the tests.**
   ```
   pip install pytest boto3
   pytest tests/
   ```

## Configuration

All thresholds and resource names are passed to the Lambda function as
environment variables (see `src/lambda/lambda_orchestrator.py` docstring).
Adjust `WARN_THRESHOLD` and `ACT_THRESHOLD` based on the precision and
recall tradeoff observed during evaluation, described in the accompanying
report.
