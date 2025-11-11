import type { RatioDefinition } from "../api/types";

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
