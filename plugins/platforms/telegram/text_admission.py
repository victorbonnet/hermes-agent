"""Serialize built-in text preparation and handoff per bot/chat/message.

PTB block=False handlers outlive process_update and the per-chat processor.
Locks pin all waiters, so cancellation never creates a second lock for a live key.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

from gateway.platforms.event import MessageEvent, MessageType


@asynccontextmanager
async def _message_turn(adapter, key):
    entries = adapter._text_message_locks
    entry = entries.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            del entries[key]


def serialized_text(callback):
    @wraps(callback)
    async def serialized(adapter, update, context):
        admission = getattr(adapter, "_update_admission", None)
        claim = admission.get() if admission is not None else None
        message = update.effective_message
        if claim is None or message is None:
            return await callback(adapter, update, context)
        key = (claim.key.split(":", 1)[0], message.chat_id, message.message_id)
        async with _message_turn(adapter, key):
            # Content comparison happens inside callback, after its auth/trigger gate,
            # and only after the predecessor has either handed off or abandoned work.
            return await callback(adapter, update, context)
    return serialized


class TelegramTextAdmissionMixin:
    async def _build_triggered_event(self, msg, update, msg_type: MessageType) -> MessageEvent:
        """Event for an addressed text/command: trigger text cleaned (sole addressee only), replied-to
        media cached, attribution applied."""
        from plugins.platforms.telegram.telegram_context import group_trigger_text
        event = self._build_message_event(msg, msg_type, update_id=update.update_id)
        event.text = group_trigger_text(self, msg, event.text)
        await self._cache_replied_media(msg, event)
        return self._apply_telegram_group_observe_attribution(event)

    @serialized_text
    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text; buffers client-split chunks into one MessageEvent."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        # Auth check first: blocked users must not reach batching, the observed transcript, or the agent.
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, MessageType.TEXT):
            return
        from .update_admission import unchanged_text
        if unchanged_text(self, update):
            return
        await self._ensure_forum_commands(update.message)
        self._enqueue_text_event(await self._build_triggered_event(msg, update, MessageType.TEXT))

    @serialized_text
    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        if not self._should_process_message(msg, is_command=True):
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        from .update_admission import unchanged_text
        if unchanged_text(self, update):
            return
        await self._ensure_forum_commands(msg)
        event = await self._build_triggered_event(msg, update, MessageType.COMMAND)
        # A >4096-char command paste arrives as a near-limit COMMAND chunk plus TEXT continuations; dispatching
        # immediately would orphan them. Near-limit commands go through text batching.
        if len(event.text or "") >= self._SPLIT_THRESHOLD:
            self._enqueue_text_event(event)
            return
        await self.handle_message(event)
