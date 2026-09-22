"""
AWS Glue job: transforms raw CloudWatch metrics delivered to the S3 data
lake into hourly feature rows used for both training and labeling.

Reads:  s3://<bucket>/raw/cloudwatch-metrics/
Writes: s3://<bucket>/curated/features/  (partitioned by dt)

Run as a Glue ETL job (Glue 4.0, Python 3, G.1X workers). Scheduled daily
by EventBridge to keep the curated feature set current for the weekly
SageMaker training pipeline.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ["JOB_NAME", "raw_path", "curated_path", "health_events_path"])

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

raw_metrics = spark.read.parquet(args["raw_path"])

# Roll metrics up to hourly windows per instance
hourly = (
    raw_metrics
    .withColumn("hour", F.date_trunc("hour", F.col("timestamp")))
    .groupBy("instance_id", "hour")
    .agg(
        F.avg("cpu_utilization").alias("cpu_util_mean"),
        F.max("cpu_utilization").alias("cpu_util_max"),
        F.stddev("cpu_utilization").alias("cpu_util_std"),
        F.avg("disk_read_bytes").alias("disk_io_mean"),
        F.max("disk_read_bytes").alias("disk_io_max"),
        F.avg("network_out").alias("network_out_mean"),
        F.stddev("network_out").alias("network_out_std"),
        F.sum("status_check_failed").alias("status_check_failed_count"),
        F.avg("mem_used_percent").alias("memory_util_mean"),
    )
    .na.fill(0.0)
)

# Bring in confirmed failure and replacement events (from AWS Health and
# Systems Manager runbook outcomes) to build the supervised label.
health_events = spark.read.parquet(args["health_events_path"]).select(
    "instance_id", F.col("event_time").alias("failure_time")
)

window_spec = Window.partitionBy("instance_id").orderBy("hour")
labeled = (
    hourly.join(health_events, on="instance_id", how="left")
    .withColumn(
        "hours_to_failure",
        (F.col("failure_time").cast("long") - F.col("hour").cast("long")) / 3600,
    )
    .withColumn(
        "label",
        F.when(
            (F.col("hours_to_failure") >= 0) & (F.col("hours_to_failure") <= 4), 1
        ).otherwise(0),
    )
    .groupBy(
        "instance_id", "hour", "cpu_util_mean", "cpu_util_max", "cpu_util_std",
        "disk_io_mean", "disk_io_max", "network_out_mean", "network_out_std",
        "status_check_failed_count", "memory_util_mean",
    )
    .agg(F.max("label").alias("label"))
    .withColumn("dt", F.to_date("hour"))
)

output_columns = [
    "label", "cpu_util_mean", "cpu_util_max", "cpu_util_std",
    "disk_io_mean", "disk_io_max", "network_out_mean", "network_out_std",
    "status_check_failed_count", "memory_util_mean", "dt",
]

(
    labeled.select(*output_columns)
    .repartition("dt")
    .write.mode("overwrite")
    .partitionBy("dt")
    .csv(args["curated_path"], header=False)
)

job.commit()
