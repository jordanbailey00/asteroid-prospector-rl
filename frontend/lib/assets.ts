export interface GraphicsFrameDef {
  atlas: string;
  frame: string;
  path?: string;
  color?: string;
  shape?: string;
  glyph?: string;
}

export interface GraphicsManifest {
  version: string;
  atlases: Record<string, { image: string; data: string }>;
  frames: Record<string, GraphicsFrameDef>;
  backgrounds: Record<string, { path: string; type: string }>;
  fonts: Record<string, string>;
  colors: Record<string, string>;
}

export interface AudioGroupDef {
  basePath: string;
  volume: number;
}

export interface AudioCueDef {
  group: string;
  files: string[];
  synth: string;
  volume?: number;
}

export interface AudioManifest {
  version: string;
  groups: Record<string, AudioGroupDef>;
  cues: Record<string, AudioCueDef>;
}

const GRAPHICS_MANIFEST_URL = "/assets/manifests/graphics_manifest.json";
const AUDIO_MANIFEST_URL = "/assets/manifests/audio_manifest.json";

const DEFAULT_GRAPHICS_MANIFEST: GraphicsManifest = {
  version: "m6.5.fallback",
  atlases: {},
  frames: {},
  backgrounds: {
    "bg.starfield.0": {
      path: "/assets/backgrounds/starfield.svg",
      type: "svg",
    },
  },
  fonts: {},
  colors: {},
};

const DEFAULT_AUDIO_MANIFEST: AudioManifest = {
  version: "m6.5.fallback",
  groups: {
    ui: { basePath: "/assets/audio/ui/", volume: 0.6 },
    sfx: { basePath: "/assets/audio/sfx/", volume: 0.8 },
  },
  cues: {
    "ui.none": { group: "ui", files: [], synth: "none" },
    "sfx.none": { group: "sfx", files: [], synth: "none" },
  },
};

let graphicsManifestPromise: Promise<GraphicsManifest> | null = null;
let audioManifestPromise: Promise<AudioManifest> | null = null;

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`failed to fetch manifest: ${url}`);
  }
  return (await response.json()) as T;
}

export async function loadGraphicsManifest(): Promise<GraphicsManifest> {
  if (!graphicsManifestPromise) {
    graphicsManifestPromise = fetchJson<GraphicsManifest>(GRAPHICS_MANIFEST_URL).catch(
      () => DEFAULT_GRAPHICS_MANIFEST,
    );
  }
  return graphicsManifestPromise;
}

export async function loadAudioManifest(): Promise<AudioManifest> {
  if (!audioManifestPromise) {
    audioManifestPromise = fetchJson<AudioManifest>(AUDIO_MANIFEST_URL).catch(
      () => DEFAULT_AUDIO_MANIFEST,
    );
  }
  return audioManifestPromise;
}

export function resetManifestCacheForTests(): void {
  graphicsManifestPromise = null;
  audioManifestPromise = null;
}
