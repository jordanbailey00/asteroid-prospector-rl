"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AudioManifest, loadAudioManifest } from "@/lib/assets";

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function playSynth(
  context: AudioContext,
  synth: string,
  volume: number,
): void {
  if (synth === "none") {
    return;
  }

  const now = context.currentTime;
  const gain = context.createGain();
  gain.connect(context.destination);
  gain.gain.setValueAtTime(0.0001, now);

  const osc = context.createOscillator();
  osc.connect(gain);

  let duration = 0.12;
  let startFreq = 280;
  let endFreq = 280;
  let type: OscillatorType = "sine";

  switch (synth) {
    case "ui_play":
      type = "triangle";
      startFreq = 440;
      endFreq = 660;
      duration = 0.14;
      break;
    case "ui_pause":
      type = "square";
      startFreq = 360;
      endFreq = 260;
      duration = 0.11;
      break;
    case "ui_step":
      type = "triangle";
      startFreq = 520;
      endFreq = 500;
      duration = 0.07;
      break;
    case "scan":
      type = "sine";
      startFreq = 720;
      endFreq = 980;
      duration = 0.22;
      break;
    case "laser":
      type = "sawtooth";
      startFreq = 1100;
      endFreq = 480;
      duration = 0.1;
      break;
    case "warp":
      type = "sawtooth";
      startFreq = 240;
      endFreq = 90;
      duration = 0.18;
      break;
    case "alarm":
      type = "square";
      startFreq = 880;
      endFreq = 640;
      duration = 0.24;
      break;
    case "alarm_short":
      type = "square";
      startFreq = 720;
      endFreq = 520;
      duration = 0.1;
      break;
    case "hiss":
      type = "triangle";
      startFreq = 220;
      endFreq = 180;
      duration = 0.2;
      break;
    case "explosion":
      type = "sawtooth";
      startFreq = 180;
      endFreq = 70;
      duration = 0.3;
      break;
    case "engine_fail":
      type = "square";
      startFreq = 240;
      endFreq = 110;
      duration = 0.26;
      break;
    case "run_end":
      type = "triangle";
      startFreq = 420;
      endFreq = 740;
      duration = 0.34;
      break;
    case "cash":
      type = "triangle";
      startFreq = 640;
      endFreq = 920;
      duration = 0.16;
      break;
    case "confirm":
      type = "triangle";
      startFreq = 580;
      endFreq = 680;
      duration = 0.1;
      break;
    case "utility":
      type = "sine";
      startFreq = 500;
      endFreq = 440;
      duration = 0.14;
      break;
    default:
      type = "sine";
      startFreq = 420;
      endFreq = 400;
      duration = 0.1;
      break;
  }

  osc.type = type;
  osc.frequency.setValueAtTime(startFreq, now);
  osc.frequency.linearRampToValueAtTime(endFreq, now + duration);

  const finalVolume = clamp(volume, 0, 1);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.001, finalVolume), now + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc.start(now);
  osc.stop(now + duration + 0.02);
}

export function useCuePlayer(defaultEnabled = false) {
  const [enabled, setEnabled] = useState(defaultEnabled);
  const manifestRef = useRef<AudioManifest | null>(null);
  const contextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    void loadAudioManifest().then((manifest) => {
      manifestRef.current = manifest;
    });
  }, []);

  useEffect(() => {
    return () => {
      if (contextRef.current) {
        void contextRef.current.close();
      }
    };
  }, []);

  const playCue = useCallback(
    async (cueKey: string) => {
      if (!enabled || cueKey.trim() === "") {
        return;
      }

      const manifest = manifestRef.current ?? (await loadAudioManifest());
      manifestRef.current = manifest;

      const cue = manifest.cues[cueKey];
      if (!cue) {
        return;
      }

      const group = manifest.groups[cue.group] ?? {
        basePath: "",
        volume: 1,
      };
      const volume = clamp((group.volume ?? 1) * (cue.volume ?? 1), 0, 1);

      if (cue.files.length > 0) {
        const file = cue.files[Math.floor(Math.random() * cue.files.length)];
        const src = `${group.basePath}${file}`;
        const audio = new Audio(src);
        audio.volume = volume;
        void audio.play().catch(() => {
          // Browser autoplay policies can block playback until user gesture.
        });
        return;
      }

      const ContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!ContextCtor) {
        return;
      }

      let audioContext = contextRef.current;
      if (!audioContext) {
        audioContext = new ContextCtor();
        contextRef.current = audioContext;
      }

      if (audioContext.state === "suspended") {
        await audioContext.resume().catch(() => {
          // Resume can fail if tab is not active.
        });
      }

      playSynth(audioContext, cue.synth, volume);
    },
    [enabled],
  );

  return {
    enabled,
    setEnabled,
    playCue,
  };
}
