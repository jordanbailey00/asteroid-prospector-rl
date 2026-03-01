import {
  MetricsWindowsResponse,
  PlaySessionCreateRequest,
  PlaySessionDeleteResponse,
  PlaySessionResetRequest,
  PlaySessionState,
  PlaySessionStepRequest,
  ReplayEntry,
  ReplayFrame,
  ReplayFramesResponse,
  ReplayListResponse,
  RunDetailResponse,
  RunsResponse,
} from "@/lib/types";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/$/, "");
}

function deriveWsBase(httpBase: string): string {
  if (httpBase.startsWith("https://")) {
    return `wss://${httpBase.slice("https://".length)}`;
  }
  if (httpBase.startsWith("http://")) {
    return `ws://${httpBase.slice("http://".length)}`;
  }
  return httpBase;
}

const HTTP_BASE = trimTrailingSlash(
  process.env.NEXT_PUBLIC_BACKEND_HTTP_BASE || "http://127.0.0.1:8000",
);

const WS_BASE = trimTrailingSlash(
  process.env.NEXT_PUBLIC_BACKEND_WS_BASE || deriveWsBase(HTTP_BASE),
);

function buildUrl(path: string, query?: Record<string, string | number | undefined>): string {
  const url = new URL(`${HTTP_BASE}${path}`);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined) {
        continue;
      }
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

function buildWsUrl(path: string, query?: Record<string, string | number | undefined>): string {
  const url = new URL(`${WS_BASE}${path}`);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined) {
        continue;
      }
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

async function requestJson<T>(
  path: string,
  options?: RequestInit,
  query?: Record<string, string | number | undefined>,
): Promise<T> {
  const response = await fetch(buildUrl(path, query), {
    cache: "no-store",
    ...options,
    headers: {
      "content-type": "application/json",
      ...(options?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = "request failed";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      detail = await response.text();
    }
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }

  return (await response.json()) as T;
}

export function backendBaseUrl(): string {
  return HTTP_BASE;
}

export function backendWsBase(): string {
  return WS_BASE;
}

export async function listRuns(limit = 50): Promise<RunsResponse> {
  return requestJson<RunsResponse>("/api/runs", undefined, { limit });
}

export async function getRun(runId: string): Promise<RunDetailResponse> {
  return requestJson<RunDetailResponse>(`/api/runs/${runId}`);
}

export async function listReplays(
  runId: string,
  options?: {
    tag?: string;
    tagsAny?: string;
    tagsAll?: string;
    windowId?: number;
    limit?: number;
  },
): Promise<ReplayListResponse> {
  return requestJson<ReplayListResponse>(`/api/runs/${runId}/replays`, undefined, {
    tag: options?.tag,
    tags_any: options?.tagsAny,
    tags_all: options?.tagsAll,
    window_id: options?.windowId,
    limit: options?.limit ?? 100,
  });
}

export async function getReplay(runId: string, replayId: string): Promise<ReplayEntry> {
  const payload = await requestJson<{ run_id: string; replay: ReplayEntry }>(
    `/api/runs/${runId}/replays/${replayId}`,
  );
  return payload.replay;
}

export async function getReplayFrames(
  runId: string,
  replayId: string,
  options?: { offset?: number; limit?: number },
): Promise<ReplayFramesResponse> {
  return requestJson<ReplayFramesResponse>(
    `/api/runs/${runId}/replays/${replayId}/frames`,
    undefined,
    {
      offset: options?.offset ?? 0,
      limit: options?.limit ?? 1024,
    },
  );
}

type ReplayFramesWsMessage = {
  type?: string;
  detail?: string;
  status_code?: number;
  next_offset?: number;
  has_more?: boolean;
  frames?: ReplayFrame[];
};

export async function getReplayFramesWebSocket(
  runId: string,
  replayId: string,
  options?: {
    offset?: number;
    limit?: number;
    batchSize?: number;
    maxChunkBytes?: number;
    yieldEveryBatches?: number;
  },
): Promise<ReplayFramesResponse> {
  if (typeof WebSocket === "undefined") {
    throw new Error("WebSocket replay transport is unavailable in this runtime");
  }

  const offset = options?.offset ?? 0;
  const limit = options?.limit ?? 1024;
  const batchSize = options?.batchSize ?? 256;
  const maxChunkBytes = options?.maxChunkBytes;
  const yieldEveryBatches = options?.yieldEveryBatches;

  const wsUrl = buildWsUrl(
    `/ws/runs/${encodeURIComponent(runId)}/replays/${encodeURIComponent(replayId)}/frames`,
    {
      offset,
      limit,
      batch_size: batchSize,
      max_chunk_bytes: maxChunkBytes,
      yield_every_batches: yieldEveryBatches,
    },
  );

  return await new Promise<ReplayFramesResponse>((resolve, reject) => {
    const socket = new WebSocket(wsUrl);
    const frames: ReplayFrame[] = [];

    let settled = false;
    let nextOffset = offset;
    let hasMore = false;

    const fail = (message: string) => {
      if (settled) {
        return;
      }
      settled = true;
      try {
        socket.close();
      } catch {
        // no-op
      }
      reject(new Error(message));
    };

    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      resolve({
        run_id: runId,
        replay_id: replayId,
        offset,
        next_offset: nextOffset,
        count: frames.length,
        has_more: hasMore,
        frames,
      });
    };

    socket.onmessage = (event) => {
      let message: ReplayFramesWsMessage;
      try {
        message = JSON.parse(String(event.data)) as ReplayFramesWsMessage;
      } catch {
        fail("WebSocket replay stream returned invalid JSON payload");
        return;
      }

      const messageType = String(message.type ?? "");
      if (messageType === "error") {
        const detail = String(message.detail ?? "websocket replay request failed");
        fail(`${message.status_code ?? 500}: ${detail}`);
        return;
      }

      if (typeof message.next_offset === "number" && Number.isFinite(message.next_offset)) {
        nextOffset = Math.max(offset, Math.trunc(message.next_offset));
      }
      if (typeof message.has_more === "boolean") {
        hasMore = message.has_more;
      }

      if (messageType === "frames") {
        if (!Array.isArray(message.frames)) {
          fail("WebSocket replay stream returned malformed frame batch");
          return;
        }
        for (const frame of message.frames) {
          frames.push(frame);
        }
        return;
      }

      if (messageType === "complete") {
        finish();
        return;
      }

      fail(`Unexpected websocket replay message type: ${messageType || "<missing>"}`);
    };

    socket.onerror = () => {
      fail("WebSocket replay stream encountered a transport error");
    };

    socket.onclose = (event) => {
      if (settled) {
        return;
      }
      fail(`WebSocket replay stream closed before completion (code ${event.code})`);
    };
  });
}

export async function getMetricsWindows(
  runId: string,
  options?: { limit?: number; order?: "asc" | "desc" },
): Promise<MetricsWindowsResponse> {
  return requestJson<MetricsWindowsResponse>(`/api/runs/${runId}/metrics/windows`, undefined, {
    limit: options?.limit ?? 500,
    order: options?.order ?? "desc",
  });
}

export async function createPlaySession(
  body?: PlaySessionCreateRequest,
): Promise<PlaySessionState> {
  return requestJson<PlaySessionState>("/api/play/session", {
    method: "POST",
    body: JSON.stringify(body ?? {}),
  });
}

export async function resetPlaySession(
  sessionId: string,
  body?: PlaySessionResetRequest,
): Promise<PlaySessionState> {
  return requestJson<PlaySessionState>(`/api/play/session/${sessionId}/reset`, {
    method: "POST",
    body: JSON.stringify(body ?? {}),
  });
}

export async function stepPlaySession(
  sessionId: string,
  body: PlaySessionStepRequest,
): Promise<PlaySessionState> {
  return requestJson<PlaySessionState>(`/api/play/session/${sessionId}/step`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function deletePlaySession(sessionId: string): Promise<PlaySessionDeleteResponse> {
  return requestJson<PlaySessionDeleteResponse>(`/api/play/session/${sessionId}`, {
    method: "DELETE",
  });
}
