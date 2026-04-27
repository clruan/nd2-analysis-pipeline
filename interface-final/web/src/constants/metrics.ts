import type { ChannelDefinition, RatioDefinition } from "../api/types";

const DEFAULT_CHANNEL_COLORS = [
  "#00ff00",
  "#ff0000",
  "#0000ff",
  "#ffff00",
  "#00ffff",
  "#ff00ff",
  "#ffffff",
  "#ff8800"
];

const defaultChannelColor = (channel: number) => DEFAULT_CHANNEL_COLORS[(Math.max(channel, 1) - 1) % DEFAULT_CHANNEL_COLORS.length];

export const DEFAULT_RATIO_DEFINITIONS: RatioDefinition[] = [
  {
    id: "channel_1_3_ratio",
    label: "Channel 1 / Channel 3",
    numerator_channel: 1,
    denominator_channel: 3
  },
  {
    id: "channel_2_3_ratio",
    label: "Channel 2 / Channel 3",
    numerator_channel: 2,
    denominator_channel: 3
  }
];

export const buildDefaultChannelDefinitions = (channelIds?: number[]): ChannelDefinition[] => {
  const ids = (channelIds && channelIds.length ? channelIds : [1, 2, 3])
    .map((channel) => Number(channel))
    .filter((channel) => Number.isFinite(channel) && channel > 0)
    .sort((a, b) => a - b);
  const uniqueIds = Array.from(new Set(ids));
  return uniqueIds.map((channel) => ({
    channel,
    label: `Channel ${channel}`,
    color: defaultChannelColor(channel)
  }));
};

export const DEFAULT_CHANNEL_DEFINITIONS: ChannelDefinition[] = buildDefaultChannelDefinitions();

export const normalizeChannelDefinitions = (
  definitions?: ChannelDefinition[] | null,
  channelIds?: number[] | null
): ChannelDefinition[] => {
  const providedIds = (definitions ?? [])
    .map((definition) => Number(definition.channel))
    .filter((channel) => Number.isFinite(channel) && channel > 0);
  const mergedIds = Array.from(new Set([...(channelIds ?? []), ...providedIds])).sort((a, b) => a - b);
  const defaults = buildDefaultChannelDefinitions(mergedIds.length ? mergedIds : undefined);
  const merged = new Map<number, ChannelDefinition>();
  defaults.forEach((definition) => {
    merged.set(definition.channel, { ...definition });
  });
  (definitions ?? []).forEach((definition) => {
    const channel = Number(definition.channel);
    if (!Number.isFinite(channel) || channel <= 0) return;
    const fallback = merged.get(channel) ?? {
      channel,
      label: `Channel ${channel}`,
      color: defaultChannelColor(channel)
    };
    merged.set(channel, {
      channel,
      label: (definition.label ?? "").trim() || fallback.label,
      color: (definition.color ?? "").trim() || fallback.color
    });
  });
  return Array.from(merged.values()).sort((a, b) => a.channel - b.channel);
};

export const buildChannelMetrics = (definitions?: ChannelDefinition[] | null) =>
  normalizeChannelDefinitions(definitions).map((definition) => ({
    id: `channel_${definition.channel}_area`,
    label: `${definition.label} Area (%)`,
    channel: definition.channel
  }));

export const channelLabelMap = (
  definitions?: ChannelDefinition[] | null,
  channelIds?: number[] | null
): Record<number, string> => {
  const normalized = normalizeChannelDefinitions(definitions, channelIds);
  return normalized.reduce<Record<number, string>>((acc, definition) => {
    acc[definition.channel] = definition.label;
    return acc;
  }, {});
};
