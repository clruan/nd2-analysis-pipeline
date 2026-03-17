import { useMemo } from "react";
import type { ChannelRangePayload } from "../api/hooks";
import { useAppStore } from "../state/useAppStore";
import { useDebouncedValue } from "./useDebouncedValue";

type ChannelId = "channel_1" | "channel_2" | "channel_3";

const toPayload = (ranges: Record<ChannelId, [number, number]>): ChannelRangePayload => {
  const payload: ChannelRangePayload = {};
  (Object.keys(ranges) as ChannelId[]).forEach((channel) => {
    const [vmin, vmax] = ranges[channel];
    payload[channel] = { vmin, vmax };
  });
  return payload;
};

export function usePreviewRanges(debounceMs = 250) {
  const previewChannelRanges = useAppStore((state) => state.previewChannelRanges);
  const debouncedRanges = useDebouncedValue(previewChannelRanges, debounceMs);

  const payload = useMemo(() => toPayload(previewChannelRanges), [previewChannelRanges]);
  const debouncedPayload = useMemo(() => toPayload(debouncedRanges), [debouncedRanges]);

  // Signature helps cache keys detect range changes without passing the full object.
  const debouncedSignature = useMemo(() => {
    return (Object.keys(debouncedRanges) as ChannelId[])
      .map((channel) => {
        const [min, max] = debouncedRanges[channel];
        return `${channel}:${min}-${max}`;
      })
      .sort()
      .join("|");
  }, [debouncedRanges]);

  return { payload, debouncedPayload, debouncedSignature };
}
