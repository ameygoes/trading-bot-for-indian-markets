"""Notification layer — provider-agnostic alert and approval system."""

from .base import ApprovalChoice, ApprovalResponse, NotificationProvider, StockAlert
from .gateway import NotificationGateway, get_gateway

__all__ = [
    "ApprovalChoice",
    "ApprovalResponse",
    "NotificationProvider",
    "StockAlert",
    "NotificationGateway",
    "get_gateway",
]
