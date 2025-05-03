import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

class MetricsService:
    """
    Service for exposing metrics about the MX Clipping system.
    
    This service uses Prometheus to expose metrics such as clip precision and
    average latency, making them available for monitoring and alerting.
    """
    
    def __init__(self, port: int = 8000):
        """
        Initialize the metrics service.
        
        Args:
            port: Port to expose the Prometheus metrics on
        """
        self.port = port
        self.server_started = False
        
        # Counters
        self.clips_created = Counter(
            'mxclip_clips_created_total',
            'Total number of clips created',
            ['reason', 'user_id']
        )
        
        self.clips_failed = Counter(
            'mxclip_clips_failed_total',
            'Total number of clip creation failures',
            ['reason', 'user_id']
        )
        
        self.triggers_received = Counter(
            'mxclip_triggers_received_total',
            'Total number of triggers received',
            ['reason', 'user_id']
        )
        
        self.triggers_dropped = Counter(
            'mxclip_triggers_dropped_total',
            'Total number of triggers dropped',
            ['reason', 'user_id']
        )
        
        # Gauges
        self.active_users = Gauge(
            'mxclip_active_users',
            'Number of active users with processors'
        )
        
        self.queue_size = Gauge(
            'mxclip_queue_size',
            'Current size of the trigger queue',
            ['user_id']
        )
        
        # Histograms for latency measurements
        self.clip_processing_time = Histogram(
            'mxclip_clip_processing_seconds',
            'Time taken to process and create a clip',
            ['reason'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.trigger_to_clip_latency = Histogram(
            'mxclip_trigger_to_clip_seconds',
            'Time from trigger to clip creation',
            ['reason'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Moving averages for precision metrics
        self._clips_total = 0
        self._clips_relevant = 0
        self._clips_precision = Gauge(
            'mxclip_clip_precision',
            'Precision of generated clips (relevant / total)'
        )
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start(self):
        """Start the metrics server."""
        if not self.server_started:
            start_http_server(self.port)
            self.server_started = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def record_clip_created(self, reason: str, user_id: str, processing_time: float):
        """
        Record a clip creation event.
        
        Args:
            reason: Reason for the clip creation
            user_id: ID of the user who triggered the clip
            processing_time: Time taken to process and create the clip
        """
        self.clips_created.labels(reason=reason, user_id=user_id).inc()
        self.clip_processing_time.labels(reason=reason).observe(processing_time)
    
    def record_clip_failed(self, reason: str, user_id: str):
        """
        Record a clip creation failure.
        
        Args:
            reason: Reason for the failed clip creation
            user_id: ID of the user who triggered the clip
        """
        self.clips_failed.labels(reason=reason, user_id=user_id).inc()
    
    def record_trigger(self, reason: str, user_id: str, dropped: bool = False):
        """
        Record a trigger event.
        
        Args:
            reason: Reason for the trigger
            user_id: ID of the user who triggered the event
            dropped: Whether the trigger was dropped
        """
        self.triggers_received.labels(reason=reason, user_id=user_id).inc()
        if dropped:
            self.triggers_dropped.labels(reason=reason, user_id=user_id).inc()
    
    def record_trigger_to_clip_latency(self, reason: str, latency: float):
        """
        Record the latency from trigger to clip creation.
        
        Args:
            reason: Reason for the trigger
            latency: Time in seconds from trigger to clip creation
        """
        self.trigger_to_clip_latency.labels(reason=reason).observe(latency)
    
    def update_active_users(self, count: int):
        """
        Update the number of active users.
        
        Args:
            count: Number of active users
        """
        self.active_users.set(count)
    
    def update_queue_size(self, user_id: str, size: int):
        """
        Update the queue size for a user.
        
        Args:
            user_id: ID of the user
            size: Current queue size
        """
        self.queue_size.labels(user_id=user_id).set(size)
    
    def update_clip_precision(self, relevant: bool):
        """
        Update the clip precision metric.
        
        Args:
            relevant: Whether the clip was relevant (true positive)
        """
        with self.lock:
            self._clips_total += 1
            if relevant:
                self._clips_relevant += 1
            
            # Calculate precision and update gauge
            precision = self._clips_relevant / self._clips_total if self._clips_total > 0 else 0
            self._clips_precision.set(precision)
    
    def update_user_stats(self, stats: List[Dict[str, Any]]):
        """
        Update metrics based on user processor statistics.
        
        Args:
            stats: List of user processor statistics
        """
        # Update active users count
        active_count = sum(1 for stat in stats if stat.get('running', False))
        self.update_active_users(active_count)
        
        # Update queue sizes
        for stat in stats:
            if 'user_id' in stat and 'queue_size' in stat:
                self.update_queue_size(stat['user_id'], stat['queue_size'])


# Global metrics service instance for use throughout the application
metrics_service = MetricsService()


def initialize_metrics(port: int = 8000):
    """
    Initialize and start the metrics service.
    
    Args:
        port: Port to expose the Prometheus metrics on
    
    Returns:
        Initialized metrics service
    """
    global metrics_service
    metrics_service = MetricsService(port=port)
    metrics_service.start()
    return metrics_service


def get_metrics_service() -> MetricsService:
    """
    Get the global metrics service instance.
    
    Returns:
        The metrics service
    """
    return metrics_service 