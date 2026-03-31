"""Inter-agent message bus for agent collaboration.

Agents can:
- Publish messages to topics (broadcast)
- Send private messages to specific agents
- Subscribe to topics they care about
- Messages have priority levels and rate limiting
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
from queue import PriorityQueue

logger = logging.getLogger(__name__)

class Priority(IntEnum):
    """Message priority — lower number = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class Message:
    """A message on the bus."""
    topic: str
    sender: str
    payload: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    recipient: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"{self.sender}_{int(time.time() * 1000)}"
    
    def __lt__(self, other):
        """For PriorityQueue ordering."""
        return self.priority < other.priority

# Type alias for subscriber callbacks
MessageHandler = Callable[[Message], None]

class MessageBus:
    """In-memory message bus for inter-agent communication.

    Features:
    - Topic-based pub/sub
    - Private agent-to-agent messaging
    - Priority queue for message ordering
    - Rate limiting per sender
    - Message history for debugging
    """

    def __init__(
        self,
        max_messages_per_minute: int = 60,
        history_size: int = 1000,
    ):
        self.max_rate = max_messages_per_minute
        self.history_size = history_size

        # topic -> list of (subscriber_name, handler)
        self.subscribers: Dict[str, List[tuple[str, MessageBus]]] = defaultdict(list)

        # Private mailboxes: agent_name -> list of messages
        self.mailboxes: Dict[str, List[Message]] = defaultdict(list)

        # Rate limiting: sender -> list of timestamps
        self._send_times: Dict[str, List[float]] = defaultdict(list)

        # Message history
        self.history: List[Message] = []

    def subscribe(self, topic: str, subscriber: str, handler: MessageHandler) -> None:
        """Subscribe to a topic."""
        self.subscribers[topic].append((subscriber, handler))
        logger.info(f"{subscriber} subscribed to topic '{topic}'")

    def unsubscribe(self, topic: str, subscriber: str) -> None:
        """Unsubscribe from a topic."""
        self.subscribers[topic] = [
            (name, handler)
            for name, handler in self.subscribers[topic]
            if name != subscriber
        ]

    def publish(self, message: Message) -> bool:
        """Publish a message to a topic.

        Returns False if rate limited.
        """
        if not self._check_rate_limit(message.sender):
            logger.warning(f"Rate limited: {message.sender}")
            return False

        self._record_send(message.sender)
        self._add_to_history(message)

        if message.recipient:
            # Private message — deliver to mailbox
            self.mailboxes[message.recipient].append(message)
            logger.debug(
                f"Private message from {message.sender} "
                f"to {message.recipient}: {message.topic}"
            )
            return True

        # Broadcast to all subscribers of this topic
        handlers = self.subscribers.get(message.topic, [])
        for subscriber_name, handler in handlers:
            if subscriber_name == message.sender:
                continue
            try:
                handler(message)
            except Exception as e:
                logger.error(
                    f"Handler error for {subscriber_name} "
                    f"on topic {message.topic}: {e}"
                )
        return True
    
    def get_private_messages(self, agent_name: str) -> List[Message]:
        """Get and clear private messages for an agent."""
        messages = self.mailboxes.pop(agent_name, [])
        return sorted(messages, key=lambda m: m.priority)

    def broadcast_pattern(
        self,
        sender: str,
        pattern_name: str,
        pattern_data: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """Convenience: broadcast a discovered pattern."""
        msg = Message(
            topic="patterns",
            sender=sender,
            payload={"pattern": pattern_name, **pattern_data},
            priority=priority,
        )
        return self.publish(msg)

    def _check_rate_limit(self, sender: str) -> bool:
        """Check if sender is within rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old timestamps
        self._send_times[sender] = [
            t for t in self._send_times[sender] if t > window_start
        ]

        return len(self._send_times[sender]) < self.max_rate

    def _record_send(self, sender: str) -> None:
        """Record a send timestamp for rate limiting."""
        self._send_times[sender].append(time.time())

    def _add_to_history(self, message: Message) -> None:
        """Add message to history, trimming if needed."""
        self.history.append(message)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def get_history(
        self, topic: Optional[str] = None, limit: int = 50
    ) -> List[Message]:
        """Get message history, optionally filtered by topic."""
        if topic:
            filtered = [m for m in self.history if m.topic == topic]
        else:
            filtered = self.history
        return filtered[-limit:]