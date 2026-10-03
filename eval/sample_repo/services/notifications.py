from __future__ import annotations


def format_notification(task: dict) -> str:
    """Format a task-created notification using its title."""
    return "Task created: " + task["title"]


def send_email(message: str) -> str:
    """Represent delivery to the email sink."""
    return message


def notify(task: dict) -> str:
    """Format a task notification and send it by email."""
    return send_email(format_notification(task))
