import type { ChannelDefinition, RatioDefinition } from "../api/types";

export const CHANNEL_METRICS = [
  { id: "channel_1_area", label: "Channel 1 Area (%)", channel: 1 },
  { id: "channel_2_area", label: "Channel 2 Area (%)", channel: 2 },
  { id: "channel_3_area", label: "Channel 3 Area (%)", channel: 3 }
] as const;

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

export const DEFAULT_CHANNEL_DEFINITIONS: ChannelDefinition[] = [
  { channel: 1, label: "Channel 1", color: "#00ff00" },
  { channel: 2, label: "Channel 2", color: "#ff0000" },
  { channel: 3, label: "Channel 3", color: "#0000ff" }
];

export const normalizeChannelDefinitions = (
  definitions?: ChannelDefinition[] | null
): ChannelDefinition[] => {
  const merged = new Map<number, ChannelDefinition>();
  DEFAULT_CHANNEL_DEFINITIONS.forEach((definition) => {
    merged.set(definition.channel, { ...definition });
  });
  (definitions ?? []).forEach((definition) => {
    const channel = Number(definition.channel);
    if (![1, 2, 3].includes(channel)) return;
    const fallback = merged.get(channel)!;
    merged.set(channel, {
      channel,
      label: (definition.label ?? "").trim() || fallback.label,
      color: (definition.color ?? "").trim() || fallback.color
    });
  });
  return [1, 2, 3].map((channel) => merged.get(channel)!);
};

export const channelLabelMap = (definitions?: ChannelDefinition[] | null): Record<number, string> => {
  const normalized = normalizeChannelDefinitions(definitions);
  return normalized.reduce<Record<number, string>>((acc, definition) => {
    acc[definition.channel] = definition.label;
    return acc;
  }, {});
};
