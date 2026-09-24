"""Built-in local components for the synthetic equipment-maintenance triage example."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SensorFeaturesV1:
    version = "sensor_features_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        return pd.DataFrame(
            {
                "stress": np.clip(0.50 * records["temperature"] + 0.50 * records["vibration"], 0.0, 1.0),
            },
            index=records.index,
        )


class SensorFeaturesV2:
    version = "sensor_features_v2"

    def run(self, records: pd.DataFrame, inputs: dict):
        return pd.DataFrame(
            {
                "stress": np.clip(0.45 * records["temperature"] + 0.65 * records["vibration"], 0.0, 1.0),
            },
            index=records.index,
        )


class FailureRiskModelV1:
    version = "failure_risk_model_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        features = inputs["sensor_features"]
        return pd.Series(np.clip(0.85 * features["stress"], 0.0, 1.0), index=records.index, name="failure_risk")


class FailureRiskModelV2:
    version = "failure_risk_model_v2"

    def run(self, records: pd.DataFrame, inputs: dict):
        features = inputs["sensor_features"]
        return pd.Series(np.clip(0.95 * features["stress"] + 0.03, 0.0, 1.0), index=records.index, name="failure_risk")


class DowntimeModelV1:
    version = "downtime_model_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        return pd.Series(np.clip(records["downtime_hours"] / 10.0, 0.0, 1.0), index=records.index, name="downtime_risk")


class TriagePolicyV1:
    version = "triage_policy_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        risk = np.asarray(inputs["failure_risk_model"], dtype=float)
        downtime = np.asarray(inputs["downtime_model"], dtype=float)
        action = np.full(len(records), "monitor", dtype=object)
        action[(risk >= 0.45) | (downtime >= 0.50)] = "inspect"
        action[(risk >= 0.72) | (downtime >= 0.80)] = "service"
        return pd.Series(action, index=records.index, name="triage_action")


class TriagePolicyV2:
    version = "triage_policy_v2"

    def run(self, records: pd.DataFrame, inputs: dict):
        risk = np.asarray(inputs["failure_risk_model"], dtype=float)
        downtime = np.asarray(inputs["downtime_model"], dtype=float)
        action = np.full(len(records), "monitor", dtype=object)
        action[(risk >= 0.38) | (downtime >= 0.40)] = "inspect"
        action[(risk >= 0.68) | (downtime >= 0.85)] = "service"
        return pd.Series(action, index=records.index, name="triage_action")


class SafetyRulesV1:
    version = "safety_rules_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        action = np.asarray(inputs["triage_policy"], dtype=object).copy()
        action[records["safety_flag"].to_numpy(dtype=bool)] = "service"
        return pd.Series(action, index=records.index, name="safe_action")


class SafetyRulesV2:
    version = "safety_rules_v2"

    def run(self, records: pd.DataFrame, inputs: dict):
        action = np.asarray(inputs["triage_policy"], dtype=object).copy()
        force_service = records["safety_flag"].to_numpy(dtype=bool) & (records["vibration"].to_numpy(dtype=float) >= 0.75)
        action[force_service] = "service"
        return pd.Series(action, index=records.index, name="safe_action")


class FinalAction:
    version = "final_action_v1"

    def run(self, records: pd.DataFrame, inputs: dict):
        return pd.Series(np.asarray(inputs["safety_rules"], dtype=object), index=records.index, name="action")
