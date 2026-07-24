"""Official ACP SDK bridge used by providers with an ACP transport."""

from __future__ import annotations

import asyncio
import base64
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from arc_jobs import CancelledError

from ..errors import DeliveryState, FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from .base import (
    NativeResumeHandle,
    ProviderExecution,
    ProviderInput,
    ProviderTerminalKind,
    ProviderUsage,
)


class ACPRunner(Protocol):
    def run(
        self,
        *,
        provider: str,
        binary: str,
        model: str | None,
        prompt: str,
        inputs: tuple[ProviderInput, ...],
        session_id: str | None,
        idle_timeout_seconds: float,
        observer: Any,
        cancel: Any,
        env: Mapping[str, str] | None,
    ) -> ProviderExecution: ...


@dataclass
class _ACPClient:
    provider: str
    chunks: list[str]
    observer: Any
    activity: asyncio.Queue[None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        from acp.schema import AgentMessageChunk, TextContentBlock

        if self.activity.empty():
            self.activity.put_nowait(None)
        if not isinstance(update, AgentMessageChunk):
            return
        content = update.content
        if isinstance(content, TextContentBlock):
            self.chunks.append(content.text)
            self.observer.raw_event(
                {"type": "agent_message_chunk", "text": content.text}
            )

    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("session/request_permission")

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("fs/write_text_file")

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, *args: Any, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("session/elicitation/create")

    async def complete_elicitation(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from acp import RequestError

        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


@dataclass
class _DeliveryTracker:
    started: bool = False


class OfficialACPRunner:
    """Synchronous package boundary backed by the official async ACP SDK."""

    _CANCEL_POLL_SECONDS = 0.05

    def run(
        self,
        *,
        provider: str,
        binary: str,
        model: str | None,
        prompt: str,
        inputs: tuple[ProviderInput, ...],
        session_id: str | None,
        idle_timeout_seconds: float,
        observer: Any,
        cancel: Any,
        env: Mapping[str, str] | None,
    ) -> ProviderExecution:
        delivery = _DeliveryTracker()
        try:
            return asyncio.run(
                self._run(
                    provider=provider,
                    binary=binary,
                    model=model,
                    prompt=prompt,
                    inputs=inputs,
                    session_id=session_id,
                    idle_timeout_seconds=idle_timeout_seconds,
                    observer=observer,
                    cancel=cancel,
                    env=env,
                    delivery=delivery,
                )
            )
        except TimeoutError as exc:
            raise ProviderFailure(
                f"{provider} ACP session timed out.",
                category=FailureCategory.TIMEOUT,
                delivery=(
                    DeliveryState.MAY_HAVE_RUN
                    if delivery.started
                    else DeliveryState.NOT_DELIVERED
                ),
                retryable=True,
            ) from exc
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(
                f"{provider} ACP transport failed: {exc}",
                category=FailureCategory.TRANSPORT,
                delivery=(
                    DeliveryState.MAY_HAVE_RUN
                    if delivery.started
                    else DeliveryState.NOT_DELIVERED
                ),
                retryable=True,
            ) from exc

    async def _run(
        self,
        *,
        provider: str,
        binary: str,
        model: str | None,
        prompt: str,
        inputs: tuple[ProviderInput, ...],
        session_id: str | None,
        idle_timeout_seconds: float,
        observer: Any,
        cancel: Any,
        env: Mapping[str, str] | None,
        delivery: _DeliveryTracker,
    ) -> ProviderExecution:
        from acp import PROTOCOL_VERSION, spawn_agent_process
        from acp.schema import ClientCapabilities, Implementation

        chunks: list[str] = []
        activity: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        client = _ACPClient(provider, chunks, observer, activity)
        if self._cancel_requested(cancel):
            return ProviderExecution(ProviderTerminalKind.CANCELLED)
        async with spawn_agent_process(
            client,
            binary,
            "acp",
            env=None if env is None else dict(env),
        ) as (connection, _process):
            initialized = await asyncio.wait_for(
                connection.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(
                        name="arc-llm",
                        title="ARC LLM",
                        version="1.0.1",
                    ),
                ),
                timeout=idle_timeout_seconds,
            )
            self._validate_capabilities(initialized, inputs)
            if session_id is None:
                session = await asyncio.wait_for(
                    connection.new_session(cwd=os.getcwd(), mcp_servers=[]),
                    timeout=idle_timeout_seconds,
                )
                active_session_id = session.session_id
                if model and model != "default":
                    await asyncio.wait_for(
                        connection.set_config_option(
                            "model",
                            active_session_id,
                            model,
                        ),
                        timeout=idle_timeout_seconds,
                    )
            else:
                await asyncio.wait_for(
                    connection.resume_session(
                        session_id=session_id,
                        cwd=os.getcwd(),
                        mcp_servers=[],
                    ),
                    timeout=idle_timeout_seconds,
                )
                active_session_id = session_id
            handle = NativeResumeHandle(provider, active_session_id)
            try:
                observer.native_handle(handle)
            except CancelledError:
                return ProviderExecution(ProviderTerminalKind.CANCELLED)
            except Exception as exc:
                raise ProviderFailure(
                    "Unable to durably record the ACP native handle.",
                    category=FailureCategory.LOCAL_IO,
                    delivery=DeliveryState.NOT_DELIVERED,
                ) from exc
            blocks = self._content_blocks(prompt, inputs)
            if self._cancel_requested(cancel):
                return ProviderExecution(
                    ProviderTerminalKind.CANCELLED,
                    native_handle=handle,
                )
            try:
                observer.before_delivery()
            except CancelledError:
                return ProviderExecution(
                    ProviderTerminalKind.CANCELLED,
                    native_handle=handle,
                )
            except Exception as exc:
                raise ProviderFailure(
                    "Unable to durably record ACP delivery.",
                    category=FailureCategory.LOCAL_IO,
                    delivery=DeliveryState.NOT_DELIVERED,
                ) from exc
            delivery.started = True
            response = await self._prompt_with_supervision(
                connection=connection,
                session_id=active_session_id,
                blocks=blocks,
                idle_timeout_seconds=idle_timeout_seconds,
                activity=activity,
                cancel=cancel,
            )
            if response is None:
                return ProviderExecution(
                    ProviderTerminalKind.CANCELLED,
                    native_handle=handle,
                )
            usage = _usage(response)
            if response.stop_reason == "cancelled":
                return ProviderExecution(
                    ProviderTerminalKind.CANCELLED,
                    native_handle=handle,
                    usage=usage,
                )
            return ProviderExecution(
                ProviderTerminalKind.COMPLETED,
                candidates=(
                    CandidateMaterial(text="".join(chunks), terminal=True),
                ),
                native_handle=handle,
                usage=usage,
                diagnostics={"delivery_mode": "acp_content"},
            )

    @staticmethod
    def _validate_capabilities(
        initialized: Any,
        inputs: tuple[ProviderInput, ...],
    ) -> None:
        needs_image = any(item.media_type.startswith("image/") for item in inputs)
        needs_embedded_context = any(
            not item.media_type.startswith("image/") for item in inputs
        )
        agent = getattr(initialized, "agent_capabilities", None)
        prompt = getattr(agent, "prompt_capabilities", None)
        if needs_image and not bool(getattr(prompt, "image", False)):
            raise ProviderFailure(
                "ACP agent does not advertise image prompt support.",
                category=FailureCategory.INVALID_REQUEST,
                delivery=DeliveryState.NOT_DELIVERED,
                details={"code": "unsupported_input_media"},
            )
        if needs_embedded_context and not bool(
            getattr(prompt, "embedded_context", False)
        ):
            raise ProviderFailure(
                "ACP agent does not advertise embedded-context prompt support.",
                category=FailureCategory.INVALID_REQUEST,
                delivery=DeliveryState.NOT_DELIVERED,
                details={"code": "unsupported_input_media"},
            )

    async def _prompt_with_supervision(
        self,
        *,
        connection: Any,
        session_id: str,
        blocks: list[Any],
        idle_timeout_seconds: float,
        activity: asyncio.Queue[None],
        cancel: Any,
    ) -> Any | None:
        loop = asyncio.get_running_loop()
        last_activity = loop.time()
        prompt_task = asyncio.create_task(
            connection.prompt(session_id=session_id, prompt=blocks)
        )
        activity_task = asyncio.create_task(activity.get())
        try:
            while True:
                remaining = idle_timeout_seconds - (loop.time() - last_activity)
                if remaining <= 0:
                    await self._cancel_prompt(connection, session_id, prompt_task)
                    raise TimeoutError
                done, _ = await asyncio.wait(
                    {prompt_task, activity_task},
                    timeout=min(self._CANCEL_POLL_SECONDS, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if prompt_task in done:
                    return prompt_task.result()
                if activity_task in done:
                    last_activity = loop.time()
                    activity_task = asyncio.create_task(activity.get())
                if self._cancel_requested(cancel):
                    await self._cancel_prompt(connection, session_id, prompt_task)
                    return None
        finally:
            if not activity_task.done():
                activity_task.cancel()
            with suppress(asyncio.CancelledError):
                await activity_task

    @staticmethod
    async def _cancel_prompt(
        connection: Any,
        session_id: str,
        prompt_task: asyncio.Task[Any],
    ) -> None:
        with suppress(Exception):
            await connection.cancel(session_id=session_id)
        if not prompt_task.done():
            prompt_task.cancel()
        with suppress(BaseException):
            await prompt_task

    @staticmethod
    def _cancel_requested(cancel: Any) -> bool:
        try:
            cancel.raise_if_requested()
        except CancelledError:
            return True
        return False

    @staticmethod
    def _content_blocks(
        prompt: str,
        inputs: tuple[ProviderInput, ...],
    ) -> list[Any]:
        from acp import embedded_text_resource, image_block, text_block
        from acp.schema import EmbeddedResourceContentBlock

        blocks: list[Any] = [text_block(prompt)]
        for item in inputs:
            content = item.path.read_bytes()
            if item.media_type.startswith("image/"):
                blocks.append(
                    image_block(
                        base64.b64encode(content).decode("ascii"),
                        item.media_type,
                        uri=f"arc-input:{item.input_id}",
                    )
                )
                continue
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProviderFailure(
                    f"ACP text input is not UTF-8: {item.input_id}",
                    category=FailureCategory.INVALID_REQUEST,
                    delivery=DeliveryState.NOT_DELIVERED,
                    details={"code": "invalid_text_input"},
                ) from exc
            blocks.append(
                EmbeddedResourceContentBlock(
                    type="resource",
                    resource=embedded_text_resource(
                        f"arc-input:{item.input_id}",
                        text,
                        mime_type=item.media_type,
                    ),
                )
            )
        return blocks


def _usage(response: Any) -> ProviderUsage | None:
    value = getattr(response, "usage", None)
    if value is None:
        return None
    return ProviderUsage(
        input_tokens=_optional_int(getattr(value, "input_tokens", None)),
        output_tokens=_optional_int(getattr(value, "output_tokens", None)),
        cached_input_tokens=_optional_int(
            getattr(value, "cached_input_tokens", None)
        ),
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
