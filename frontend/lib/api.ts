import {
  MetricsWindowsResponse,
  PlaySessionCreateRequest,
  PlaySessionDeleteResponse,
  PlaySessionResetRequest,
  PlaySessionState,
  PlaySessionStepRequest,
  ReplayEntry,
  ReplayFramesResponse,
  ReplayListResponse,
  RunDetailResponse,
  RunsResponse,
} from "@/lib/types";

const HTTP_BASE =
  process.env.NEXT_PUBLIC_BACKEND_HTTP_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

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
