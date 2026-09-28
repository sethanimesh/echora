"""One explicit-memory API shared by cookie and native-session transports."""
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from . import memory


class MemoryRevision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    memory_revision: int = Field(ge=0)


class RememberInput(MemoryRevision):
    revision: int = Field(ge=1)


class DeleteProfileInput(MemoryRevision):
    profile_revision: int = Field(ge=1)


def router(session_fn, current_fn, *, invalidate_callback=None, request_adapter=None, prefix='/api'):
    """Construct routes; native supplies ``request_adapter=native_request``.

    Host security middleware/dependencies still apply. No endpoint grants speech
    authorization or modifies the message revision. Invalidation is centralized
    in memory mutations and also covers profile-edit source changes.
    """
    if invalidate_callback is not None:
        memory.set_invalidator(invalidate_callback)
    result = APIRouter(prefix=prefix)

    def session(request):
        return session_fn(request_adapter(request) if request_adapter else request)

    @result.post('/messages/{job_id}/remember')
    async def remember(job_id: str, body: RememberInput, request: Request):
        job = current_fn(session(request), job_id, body.revision)
        return memory.remember(job, body.memory_revision)

    @result.get('/profiles/{profile_id}/memories')
    async def memories(profile_id: str, request: Request):
        session(request)
        return memory.list_memories(profile_id)

    @result.post('/profiles/{profile_id}/memories/{memory_id}/delete')
    async def delete_memory(profile_id: str, memory_id: str, body: MemoryRevision, request: Request):
        session(request)
        return memory.delete_memories(profile_id, body.memory_revision, memory_id)

    @result.post('/profiles/{profile_id}/memories/clear')
    async def clear_memories(profile_id: str, body: MemoryRevision, request: Request):
        session(request)
        return memory.delete_memories(profile_id, body.memory_revision)

    @result.post('/profiles/{profile_id}/delete')
    async def delete_profile(profile_id: str, body: DeleteProfileInput, request: Request):
        session(request)
        return memory.delete_profile(profile_id, body.profile_revision, body.memory_revision)

    return result
