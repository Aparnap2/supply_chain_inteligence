"""
Enhanced Error Handling for Supply Chain Intelligence Workflow

This module provides advanced error handling capabilities including
exponential backoff, circuit breaker patterns, and graceful degradation.

Requirements covered: 5.4, 5.6
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Literal, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .state_management import CompleteWorkflowState, update_workflow_state


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling strategies."""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    DATA_VALIDATION = "data_validation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_LOGIC = "internal_logic"
    TIMEOUT = "timeout"


@dataclass
class ErrorContext:
    """Context information for error handling decisions."""
    step_name: str
    error_message: str
    error_type: str
    retry_count: int
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    recoverable: bool
    partial_results_available: bool
    data_quality_score: float


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    # State tracking
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    half_open_calls: int = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1


class EnhancedErrorHandler:
    """
    Enhanced error handler with circuit breakers, exponential backoff,
    and intelligent recovery strategies.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Initialize default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies for different error types."""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.RATE_LIMIT: self._handle_rate_limit_error,
            ErrorCategory.AUTHENTICATION: self._handle_auth_error,
            ErrorCategory.TIMEOUT: self._handle_timeout_error,
            ErrorCategory.EXTERNAL_SERVICE: self._handle_external_service_error,
            ErrorCategory.DATA_VALIDATION: self._handle_validation_error,
            ErrorCategory.RESOURCE_EXHAUSTION: self._handle_resource_error,
            ErrorCategory.INTERNAL_LOGIC: self._handle_internal_error
        }
    
    def classify_error(self, error_message: str, error_type: str, 
                      step_name: str) -> ErrorContext:
        """
        Classify error and create context for handling decisions.
        
        Args:
            error_message: Error message
            error_type: Type of error
            step_name: Step where error occurred
            
        Returns:
            ErrorContext with classification information
        """
        # Determine error category based on message and type
        category = self._categorize_error(error_message, error_type)
        
        # Determine severity
        severity = self._assess_severity(error_message, error_type, step_name)
        
        # Check if error is recoverable
        recoverable = self._is_recoverable(category, severity, step_name)
        
        return ErrorContext(
            step_name=step_name,
            error_message=error_message,
            error_type=error_type,
            retry_count=0,  # Will be updated by caller
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            recoverable=recoverable,
            partial_results_available=False,  # Will be updated by caller
            data_quality_score=0.0  # Will be updated by caller
        )
    
    def _categorize_error(self, error_message: str, error_type: str) -> ErrorCategory:
        """Categorize error based on message and type."""
        error_lower = error_message.lower()
        type_lower = error_type.lower()
        
        # Network-related errors
        if any(keyword in error_lower for keyword in 
               ['connection', 'network', 'timeout', 'unreachable', 'dns']):
            return ErrorCategory.NETWORK
        
        # Authentication errors
        if any(keyword in error_lower for keyword in 
               ['auth', 'unauthorized', 'forbidden', 'api key', 'token']):
            return ErrorCategory.AUTHENTICATION
        
        # Rate limiting
        if any(keyword in error_lower for keyword in 
               ['rate limit', 'too many requests', '429', 'quota']):
            return ErrorCategory.RATE_LIMIT
        
        # Validation errors
        if any(keyword in error_lower for keyword in 
               ['validation', 'invalid', 'malformed', 'schema']):
            return ErrorCategory.DATA_VALIDATION
        
        # Resource exhaustion
        if any(keyword in error_lower for keyword in 
               ['memory', 'disk', 'resource', 'exhausted', 'limit']):
            return ErrorCategory.RESOURCE_EXHAUSTION
        
        # Timeout errors
        if any(keyword in error_lower for keyword in 
               ['timeout', 'timed out', 'deadline']):
            return ErrorCategory.TIMEOUT
        
        # External service errors
        if any(keyword in error_lower for keyword in 
               ['service unavailable', '503', '502', '500', 'external']):
            return ErrorCategory.EXTERNAL_SERVICE
        
        # Default to internal logic error
        return ErrorCategory.INTERNAL_LOGIC
    
    def _assess_severity(self, error_message: str, error_type: str, 
                        step_name: str) -> ErrorSeverity:
        """Assess error severity based on context."""
        error_lower = error_message.lower()
        
        # Critical errors that should stop workflow
        if any(keyword in error_lower for keyword in 
               ['critical', 'fatal', 'corruption', 'security']):
            return ErrorSeverity.CRITICAL
        
        # High severity for important steps
        if step_name in ['initialize_workflow', 'generate_final_report']:
            return ErrorSeverity.HIGH
        
        # High severity for authentication and validation
        if any(keyword in error_lower for keyword in 
               ['unauthorized', 'forbidden', 'validation failed']):
            return ErrorSeverity.HIGH
        
        # Medium severity for data collection issues
        if step_name in ['collect_web_data', 'collect_api_data']:
            return ErrorSeverity.MEDIUM
        
        # Low severity for optional components
        if step_name in ['run_crew_analysis']:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _is_recoverable(self, category: ErrorCategory, severity: ErrorSeverity, 
                       step_name: str) -> bool:
        """Determine if error is recoverable."""
        # Critical errors are generally not recoverable
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Some categories are generally recoverable
        recoverable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE
        ]
        
        if category in recoverable_categories:
            return True
        
        # Authentication errors are not recoverable without intervention
        if category == ErrorCategory.AUTHENTICATION:
            return False
        
        # Data validation errors might be recoverable depending on step
        if category == ErrorCategory.DATA_VALIDATION:
            return step_name not in ['initialize_workflow']
        
        # Resource exhaustion might be recoverable with backoff
        if category == ErrorCategory.RESOURCE_EXHAUSTION:
            return True
        
        # Internal logic errors are generally not recoverable
        return False
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    async def handle_error_with_strategy(self, error_context: ErrorContext, 
                                       state: CompleteWorkflowState) -> Dict[str, Any]:
        """
        Handle error using appropriate strategy based on context.
        
        Args:
            error_context: Error context information
            state: Current workflow state
            
        Returns:
            Recovery action recommendations
        """
        # Record error in history
        self.error_history.append(error_context)
        
        # Get recovery strategy for error category
        strategy_func = self.recovery_strategies.get(
            error_context.category,
            self._handle_generic_error
        )
        
        # Execute recovery strategy
        recovery_action = await strategy_func(error_context, state)
        
        logger.info(f"Error handling strategy applied: {recovery_action}")
        return recovery_action
    
    async def _handle_network_error(self, error_context: ErrorContext, 
                                  state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle network-related errors."""
        service_name = f"{error_context.step_name}_network"
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        if not circuit_breaker.can_execute():
            return {
                "action": "skip",
                "reason": "Circuit breaker open for network operations",
                "wait_time": circuit_breaker.recovery_timeout
            }
        
        # Apply exponential backoff
        backoff_time = min(2 ** error_context.retry_count, 300)  # Max 5 minutes
        
        return {
            "action": "retry",
            "backoff_time": backoff_time,
            "max_retries": 5,
            "circuit_breaker": service_name
        }
    
    async def _handle_rate_limit_error(self, error_context: ErrorContext, 
                                     state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle rate limiting errors."""
        # For rate limits, use longer backoff times
        backoff_time = min(60 * (2 ** error_context.retry_count), 3600)  # Max 1 hour
        
        return {
            "action": "retry",
            "backoff_time": backoff_time,
            "max_retries": 3,
            "reason": "Rate limit exceeded, applying extended backoff"
        }
    
    async def _handle_auth_error(self, error_context: ErrorContext, 
                               state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle authentication errors."""
        return {
            "action": "fail",
            "reason": "Authentication error requires manual intervention",
            "recoverable": False
        }
    
    async def _handle_timeout_error(self, error_context: ErrorContext, 
                                  state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle timeout errors."""
        # Increase timeout for retries
        return {
            "action": "retry",
            "backoff_time": 30 * (error_context.retry_count + 1),
            "max_retries": 3,
            "timeout_multiplier": 1.5
        }
    
    async def _handle_external_service_error(self, error_context: ErrorContext, 
                                           state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle external service errors."""
        service_name = f"{error_context.step_name}_external"
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        circuit_breaker.record_failure()
        
        if not circuit_breaker.can_execute():
            return {
                "action": "skip",
                "reason": "External service circuit breaker open",
                "fallback_available": error_context.partial_results_available
            }
        
        return {
            "action": "retry",
            "backoff_time": min(10 * (2 ** error_context.retry_count), 600),
            "max_retries": 3,
            "circuit_breaker": service_name
        }
    
    async def _handle_validation_error(self, error_context: ErrorContext, 
                                     state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle data validation errors."""
        if error_context.step_name == "initialize_workflow":
            return {
                "action": "fail",
                "reason": "Initial validation failed, cannot proceed",
                "recoverable": False
            }
        
        return {
            "action": "skip",
            "reason": "Validation failed, proceeding with available data",
            "data_quality_impact": True
        }
    
    async def _handle_resource_error(self, error_context: ErrorContext, 
                                   state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle resource exhaustion errors."""
        # Longer backoff for resource issues
        backoff_time = min(120 * (2 ** error_context.retry_count), 1800)  # Max 30 minutes
        
        return {
            "action": "retry",
            "backoff_time": backoff_time,
            "max_retries": 2,
            "resource_cleanup": True
        }
    
    async def _handle_internal_error(self, error_context: ErrorContext, 
                                   state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle internal logic errors."""
        if error_context.severity == ErrorSeverity.CRITICAL:
            return {
                "action": "fail",
                "reason": "Critical internal error",
                "recoverable": False
            }
        
        return {
            "action": "retry",
            "backoff_time": 5 * (error_context.retry_count + 1),
            "max_retries": 2
        }
    
    async def _handle_generic_error(self, error_context: ErrorContext, 
                                  state: CompleteWorkflowState) -> Dict[str, Any]:
        """Handle generic/unknown errors."""
        return {
            "action": "retry",
            "backoff_time": min(5 * (2 ** error_context.retry_count), 120),
            "max_retries": 3,
            "reason": "Generic error handling applied"
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count errors by category
        category_counts = {}
        severity_counts = {}
        step_counts = {}
        
        for error in self.error_history:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
            step_counts[error.step_name] = step_counts.get(error.step_name, 0) + 1
        
        # Calculate recovery rate
        recoverable_errors = sum(1 for error in self.error_history if error.recoverable)
        recovery_rate = recoverable_errors / len(self.error_history) if self.error_history else 0
        
        return {
            "total_errors": len(self.error_history),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "step_distribution": step_counts,
            "recovery_rate": recovery_rate,
            "circuit_breaker_states": {
                name: breaker.state 
                for name, breaker in self.circuit_breakers.items()
            }
        }