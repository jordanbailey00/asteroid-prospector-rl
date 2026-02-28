import actionEffectsManifest from "@/lib/action_effects_manifest.json";

export interface ActionEffect {
  id: number;
  vfx: string;
  cue: string;
}

export interface EventEffect {
  vfx: string;
  cue: string;
}

const ACTION_COUNT = 69;

const ACTION_EFFECTS: ActionEffect[] = (actionEffectsManifest.actions as ActionEffect[])
  .slice()
  .sort((left, right) => left.id - right.id);

if (ACTION_EFFECTS.length !== ACTION_COUNT) {
  throw new Error(`expected ${ACTION_COUNT} action effect mappings`);
}

for (let index = 0; index < ACTION_EFFECTS.length; index += 1) {
  if (ACTION_EFFECTS[index].id !== index) {
    throw new Error(`action effect mapping missing id ${index}`);
  }
}

const EVENT_EFFECTS: Record<string, EventEffect> = actionEffectsManifest.events as Record<
  string,
  EventEffect
>;

export const CORE_GRAPHICS_KEYS: string[] = [
  "entity.ship.agent",
  "entity.ship.human",
  "entity.station",
  "entity.asteroid.small",
  "entity.asteroid.medium",
  "entity.asteroid.large",
  "entity.hazard.radiation",
  "entity.pirate.marker",
  "ui.panel.stats",
  "ui.panel.cargo",
  "ui.panel.market",
  "ui.panel.events",
  "ui.panel.minimap",
  "ui.button.primary",
  "ui.button.secondary",
  "ui.button.danger",
  "icon.market.up",
  "icon.market.down",
  "icon.market.flat",
  "icon.event.pirate",
  "icon.event.fracture",
  "icon.event.overheat",
  "icon.event.stranded",
  "icon.event.destroyed",
  "icon.event.invalid",
];

export const CORE_AUDIO_CUES: string[] = [
  "ui.none",
  "ui.click",
  "ui.select",
  "ui.play",
  "ui.pause",
  "ui.step",
  "ui.error",
  "sfx.none",
  "sfx.travel.warp",
  "sfx.travel.boost",
  "sfx.scan.wide",
  "sfx.scan.focus",
  "sfx.scan.deep",
  "sfx.scan.listen",
  "sfx.mine.low",
  "sfx.mine.mid",
  "sfx.mine.high",
  "sfx.utility.stabilize",
  "sfx.utility.refine",
  "sfx.utility.cooldown",
  "sfx.utility.tool_repair",
  "sfx.utility.hull_patch",
  "sfx.utility.jettison",
  "sfx.station.dock",
  "sfx.station.sell",
  "sfx.station.buy",
  "sfx.station.overhaul",
  "sfx.ui.run_end",
  "sfx.event.pirate_alarm",
  "sfx.event.explosion",
  "sfx.event.overheat_alarm",
  "sfx.event.engine_fail",
  "sfx.event.big_explosion",
];

export function actionVfxKey(action: number): string {
  if (!Number.isInteger(action) || action < 0 || action >= ACTION_EFFECTS.length) {
    return "vfx.none";
  }
  return ACTION_EFFECTS[action].vfx;
}

export function actionCueKey(action: number): string {
  if (!Number.isInteger(action) || action < 0 || action >= ACTION_EFFECTS.length) {
    return "sfx.none";
  }
  return ACTION_EFFECTS[action].cue;
}

export function eventVfxKey(eventName: string): string {
  return EVENT_EFFECTS[eventName]?.vfx ?? "vfx.none";
}

export function eventCueKey(eventName: string): string {
  return EVENT_EFFECTS[eventName]?.cue ?? "sfx.none";
}

export function allMappedVfxKeys(): string[] {
  const keys = new Set<string>(CORE_GRAPHICS_KEYS);
  for (const item of ACTION_EFFECTS) {
    keys.add(item.vfx);
  }
  for (const item of Object.values(EVENT_EFFECTS)) {
    keys.add(item.vfx);
  }
  return Array.from(keys.values());
}

export function allMappedCueKeys(): string[] {
  const keys = new Set<string>(CORE_AUDIO_CUES);
  for (const item of ACTION_EFFECTS) {
    keys.add(item.cue);
  }
  for (const item of Object.values(EVENT_EFFECTS)) {
    keys.add(item.cue);
  }
  return Array.from(keys.values());
}

export function actionEffects(): ActionEffect[] {
  return ACTION_EFFECTS.slice();
}

export function eventEffects(): Record<string, EventEffect> {
  return { ...EVENT_EFFECTS };
}
