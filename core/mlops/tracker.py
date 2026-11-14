"""MLflow lifecycle tracking, metric logging, and artifact management."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import mlflow

from backend.config import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Provides structured experiment and inference metric tracking via MLflow."""

    def __init__(
        self,
        experiment_name: str = "StyleForge-Production",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI or "file:./data/mlruns"
        self._is_available = False
        self._init_client()

    def _init_client(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._is_available = True
            logger.info("MLflowTracker connected to %s (experiment: %s)", self.tracking_uri, self.experiment_name)
        except Exception as e:
            logger.warning("MLflow server connection failed: %s. Metric logging will be skipped.", e)
            self._is_available = False

    def log_inference_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Path]] = None,
    ) -> Optional[str]:
        """Logs an evaluation or batch inference run to MLflow."""
        if not self._is_available:
            return None

        try:
            with mlflow.start_run(run_name=run_name) as run:
                # Log hyperparameters
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Log quantitative metrics
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))

                # Log artifact files
                if artifacts:
                    for _, art_path in artifacts.items():
                        if art_path.exists():
                            mlflow.log_artifact(str(art_path))

                return run.info.run_id
        except Exception as e:
            logger.warning("Failed to record run to MLflow: %s", e)
            return None
