"""Unchanged Telegram edits do not become additional model turns."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Update
from telegram.ext import Defaults

from tests.plugins.test_telegram_update_admission import connected, update


async def dispatch(adapter, app, *, uid, text="hello", edited=False, chat=42, message_id=472, kind="text", group=False, entities=None):
    payload = update(app.bot, uid, kind, text=text, edited=edited, chat=chat, group=group).to_dict()
    message = payload["edited_message" if edited else "message"]
    message["message_id"] = message_id
    if entities is not None:
        message["entities"] = entities
    await app.process_update(Update.de_json(payload, app.bot))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())


async def persist_receipts(adapter):
    # Disconnect has an intentionally short deadline; restart tests need the actual write.
    writer = adapter._update_receipt_flush
    assert writer is not None
    await asyncio.wait_for(asyncio.shield(writer), 15)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "command"])
async def test_unchanged_edit_keeps_observers_not_model_turns(monkeypatch, kind):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        observer = adapter._platform_event_handler = AsyncMock()
        await dispatch(adapter, app, uid=10, kind=kind)
        await dispatch(adapter, app, uid=11, kind=kind, edited=True)
        assert len(delivered) == 1
        assert observer.await_count == 1
        assert observer.call_args.args[0]["event_type"] == "message_edited"
        await dispatch(adapter, app, uid=12, kind=kind, edited=True)
        assert len(delivered) == 1
        assert observer.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", ["text", "text_link", "text_mention"])
async def test_semantic_changes_and_same_second_reversions_are_genuine_edits(monkeypatch, semantic):
    async with connected(monkeypatch) as (adapter, app, delivered):
        for uid, target in enumerate((1, 2, 2, 1, 1), 10):
            text, entities = str(target), []
            if semantic != "text":
                text = "hello"
                entity = {"type": semantic, "offset": 0, "length": 5}
                if semantic == "text_link":
                    entity["url"] = f"https://example.org/{target}"
                else:
                    entity["user"] = {"id": target, "is_bot": False, "first_name": f"Name {uid}"}
                entities.append(entity)
            await dispatch(adapter, app, uid=uid, text=text, edited=uid != 10, entities=entities)
        assert len(delivered) == 3
        assert len({event.edit_revision for event in delivered[1:]}) == 2


@pytest.mark.asyncio
async def test_identical_text_in_distinct_messages_chats_and_bots_is_independent(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10)
        await dispatch(adapter, app, uid=11, edited=True)
        await dispatch(adapter, app, uid=12, message_id=473)
        await dispatch(adapter, app, uid=13, chat=43, edited=True)
        assert len(delivered) == 3
        async with connected(monkeypatch, bot_id=222) as (other, other_app, other_delivered):
            await dispatch(other, other_app, uid=14, edited=True)
            assert len(other_delivered) == 1


@pytest.mark.asyncio
async def test_latest_text_receipt_survives_adapter_restart(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10, text="private text")
        await persist_receipts(adapter)
        assert len(delivered) == 1
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
    payload = json.loads(receipt_path.read_text())
    assert "private text" not in json.dumps(payload)
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=11, text="private text", edited=True)
        assert not delivered
        await dispatch(adapter, app, uid=12, text="changed", edited=True)
        assert [event.text for event in delivered] == ["changed"]


@pytest.mark.asyncio
async def test_failed_text_preparation_does_not_poison_latest_receipt(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=10)
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", AsyncMock(side_effect=OSError("preparation failed")))
            await dispatch(adapter, app, uid=11, text="changed", edited=True)
        await dispatch(adapter, app, uid=12, text="changed", edited=True)
        await dispatch(adapter, app, uid=13, text="changed", edited=True)
        assert [event.text for event in delivered] == ["hello", "changed"]


@pytest.mark.asyncio
async def test_passive_chatter_still_reaches_observation(monkeypatch):
    async with connected(monkeypatch, extra={"observe_unmentioned_group_messages": True, "require_mention": True,
                                            "allowed_chats": ["42"], "group_allowed_chats": ["42"]}) as (adapter, app, delivered):
        observed = []
        monkeypatch.setattr(adapter, "_observe_unmentioned_group_message", lambda *args, **kwargs: observed.append(args[0].text))
        await dispatch(adapter, app, uid=10, group=True)
        await dispatch(adapter, app, uid=11, group=True, edited=True)
        assert not delivered
        assert observed == ["hello", "hello"]


@pytest.mark.asyncio
async def test_cold_profile_package_reload_uses_its_own_admission_helper(monkeypatch):
    from hermes_cli.plugins_loader import PluginLoaderMixin, _evict_modules
    from hermes_cli.plugins_manifest import PluginManifest
    from tests.plugins import test_telegram_update_admission as harness
    from plugins.platforms.telegram import adapter as canonical

    name = "hermes_plugins.telegram_unchanged_regression"
    manifest = PluginManifest(name="telegram", path=str(Path(canonical.__file__).parent))
    helper_modules = []
    try:
        for generation in range(2):
            package = PluginLoaderMixin()._load_directory_module(manifest, module_name=name)
            monkeypatch.setattr(harness, "TelegramAdapter", package.adapter.TelegramAdapter)
            async with connected(monkeypatch) as (adapter, app, delivered):
                assert type(app).__module__ == f"{name}.update_admission"
                await dispatch(adapter, app, uid=10 + generation, edited=bool(generation))
                assert len(delivered) == int(generation == 0)
                helper_modules.append(sys.modules[f"{name}.update_admission"])
        assert helper_modules[0] is not helper_modules[1]
    finally:
        _evict_modules(name)


@pytest.mark.asyncio
async def test_reaction_and_metadata_edit_do_not_retrigger_but_text_emoji_does(monkeypatch):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        observer = adapter._platform_event_handler = AsyncMock()
        await dispatch(adapter, app, uid=10)
        await app.process_update(update(app.bot, 11, kind="reaction"))
        assert len(delivered) == 1
        assert observer.call_args.args[0]["event_type"] == "reaction"
        await dispatch(adapter, app, uid=12, edited=True)
        assert len(delivered) == 1
        assert observer.call_args.args[0]["event_type"] == "message_edited"
        await dispatch(adapter, app, uid=13, text="hello 👍", edited=True)
        assert [event.text for event in delivered] == ["hello", "hello 👍"]
        assert observer.await_count == 3


@pytest.mark.asyncio
async def test_accepted_text_receipt_survives_post_enqueue_failure(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        enqueue = adapter._enqueue_text_event

        def enqueue_then_fail(event):
            enqueue(event)
            raise OSError("failed after handoff")

        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_enqueue_text_event", enqueue_then_fail)
            await dispatch(adapter, app, uid=10)
        await dispatch(adapter, app, uid=11, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_concurrent_observer_cannot_accept_failed_text_preparation(monkeypatch):
    import hermes_cli.lifecycle

    monkeypatch.setattr(hermes_cli.lifecycle, "has_hook", lambda name: True)
    async with connected(monkeypatch) as (adapter, app, delivered):
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        entered, observed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def prepare(*args):
            entered.set()
            await release.wait()
            raise OSError("failed before handoff")

        async def observer(*args):
            observed.set()

        adapter._platform_event_handler = observer
        with monkeypatch.context() as broken:
            broken.setattr(adapter, "_cache_replied_media", prepare)
            await app.process_update(update(app.bot, 10, edited=True))
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(observed.wait(), 2)
            release.set()
            await app.stop()
        assert not adapter._latest_text_receipts
        await app.start()
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=True))
        await dispatch(adapter, app, uid=11, edited=True)
        await dispatch(adapter, app, uid=12, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
async def test_text_receipts_are_bounded_and_eviction_allows_unknown_edit(monkeypatch):
    from plugins.platforms.telegram import update_admission

    monkeypatch.setattr(update_admission, "_SEEN_CAP", 2)
    async with connected(monkeypatch) as (adapter, app, delivered):
        for message_id in (472, 473, 474):
            await dispatch(adapter, app, uid=message_id, message_id=message_id)
        await persist_receipts(adapter)
        assert len(adapter._latest_text_receipts) == 2
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
    assert len(json.loads(receipt_path.read_text())["message_texts"]) == 2
    async with connected(monkeypatch) as (adapter, app, delivered):
        await dispatch(adapter, app, uid=475, message_id=474, edited=True)
        assert not delivered
        await dispatch(adapter, app, uid=476, message_id=472, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "command"])
@pytest.mark.parametrize("outcome", ["accepted", "failed", "cancelled"])
async def test_nonblocking_preparation_serializes_and_rechecks(monkeypatch, kind, outcome):
    async with connected(monkeypatch) as (adapter, app, delivered):
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        entered, release = asyncio.Event(), asyncio.Event()
        preparations = []

        async def prepare(*args):
            preparations.append(asyncio.current_task())
            if len(preparations) == 1:
                entered.set()
                await release.wait()
                if outcome == "failed":
                    raise OSError("preparation failed")

        with monkeypatch.context() as preparing:
            preparing.setattr(adapter, "_cache_replied_media", prepare)
            await app.process_update(update(app.bot, 10, kind=kind))
            await asyncio.wait_for(entered.wait(), 2)
            await app.process_update(update(app.bot, 11, kind=kind, edited=True))
            # A ready-queue barrier lets PTB enter the scheduled edit callback.
            barrier = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(barrier.set_result, None)
            await barrier
            try:
                assert len(preparations) == 1, "edit must wait for unaccepted preparation"
            finally:
                if outcome == "cancelled":
                    preparations[0].cancel()
                release.set()
                await app.stop()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert len(delivered) == 1
        assert len(preparations) == (1 if outcome == "accepted" else 2)
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: not adapter._inflight_update_ids)
        assert not adapter._text_message_locks
        await app.start()
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=True))
        await dispatch(adapter, app, uid=12, kind=kind, edited=True)
        assert len(delivered) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hold", "expired-memory", "expired-restart", "preupgrade", "profile"])
async def test_receipt_lifetime_and_handoff_scope(monkeypatch, tmp_path, mode):
    import time
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from plugins.platforms.telegram import update_admission

    async with connected(monkeypatch) as (adapter, app, delivered):
        if mode == "hold":
            adapter._mark_disconnected()
        await dispatch(adapter, app, uid=10)
        await persist_receipts(adapter)
        receipt_path = adapter._update_receipt_dir / "telegram_update_receipts_111.json"
        if mode == "hold":
            await dispatch(adapter, app, uid=11, edited=True)
            assert len(adapter._held_inbound_events) == 1
            assert not delivered
            adapter._mark_connected()
            await adapter._held_inbound_redispatch_task
            assert len(delivered) == 1
            return
        if mode == "expired-memory":
            now = time.time()
            with monkeypatch.context() as clock:
                clock.setattr(time, "time", lambda: now + update_admission.RECEIPT_TTL_SECONDS + 1)
                await dispatch(adapter, app, uid=11, edited=True)
            assert len(delivered) == 2
            return
    if mode in ("expired-restart", "preupgrade"):
        payload = json.loads(receipt_path.read_text())
        if mode == "preupgrade":
            payload.pop("message_texts")
        else:
            for value in payload["message_texts"].values():
                value[1] -= update_admission.RECEIPT_TTL_SECONDS + 1
        receipt_path.write_text(json.dumps(payload))
    token = set_hermes_home_override(tmp_path / "other-profile") if mode == "profile" else None
    try:
        async with connected(monkeypatch) as (adapter, app, delivered):
            await dispatch(adapter, app, uid=11, edited=True)
            await dispatch(adapter, app, uid=12, edited=True)
            assert len(delivered) == 1  # unknown baselines fail open once, then suppress
    finally:
        if token is not None:
            reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_accepted_core_is_visible_while_native_observer_is_pending(monkeypatch):
    from telegram.ext import TypeHandler
    from tests.waiting import async_wait_until

    async with connected(monkeypatch) as (adapter, app, delivered):
        assert app is not None
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        observed, release = asyncio.Event(), asyncio.Event()

        async def observer(event, context):
            if event.update_id == 10:
                observed.set()
                await release.wait()

        app.add_handler(TypeHandler(Update, observer), group=101)
        try:
            await app.process_update(update(app.bot, 10))
            await asyncio.wait_for(observed.wait(), 2)
            await async_wait_until(lambda: bool(adapter._latest_text_receipts) and not adapter._text_message_locks)
            assert "111:10" in adapter._inflight_update_ids
            await app.process_update(update(app.bot, 11, edited=True))
            await async_wait_until(lambda: "111:11" not in adapter._inflight_update_ids)
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
            assert len(delivered) == 1
            assert "111:10" in adapter._inflight_update_ids
        finally:
            release.set()
            await app.stop()


@pytest.mark.asyncio
async def test_cancelled_middle_waiter_does_not_replace_live_message_lock(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        assert app is not None
        monkeypatch.setattr(app.bot, "_defaults", Defaults(block=False))
        entered, release = asyncio.Event(), asyncio.Event()
        preparations, middle_tasks = [], []
        create_task = app.create_task

        def capture_task(self, coroutine, update=None, **kwargs):
            task = create_task(coroutine, update=update, **kwargs)
            if update is not None and update.update_id == 11:
                middle_tasks.append(task)
            return task

        async def prepare(*args):
            preparations.append(asyncio.current_task())
            entered.set()
            await release.wait()

        async def ready_queue_barrier():
            ready = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(ready.set_result, None)
            await ready

        with monkeypatch.context() as preparing:
            preparing.setattr(type(app), "create_task", capture_task)
            preparing.setattr(adapter, "_cache_replied_media", prepare)
            try:
                await app.process_update(update(app.bot, 10))
                await asyncio.wait_for(entered.wait(), 2)
                await app.process_update(update(app.bot, 11, edited=True))
                await ready_queue_barrier()
                assert middle_tasks
                for task in middle_tasks:
                    task.cancel()
                await asyncio.gather(*middle_tasks, return_exceptions=True)
                await app.process_update(update(app.bot, 12, edited=True))
                await ready_queue_barrier()
                assert len(preparations) == 1
            finally:
                release.set()
                await app.stop()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert len(delivered) == 1
        assert not adapter._text_message_locks
