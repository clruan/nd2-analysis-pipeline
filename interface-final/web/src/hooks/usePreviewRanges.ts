import { useMemo } from "react";
import type { ChannelRangePayload } from "../api/hooks";
import { useAppStore } from "../state/useAppStore";
import { useDebouncedValue } from "./useDebouncedValue";

const toPayload = (ranges: Record<string, [number, number]>): ChannelRangePayload => {
  const payload: ChannelRangePayload = {};
  Object.entries(ranges).forEach(([channel, [vmin, vmax]]) => {
    payload[channel] = { vmin, vmax };
  });
  return payload;
};

export function usePreviewRanges(debounceMs = 250) {
  const previewChannelRanges = useAppStore((state) => state.previewChannelRanges);
  const debouncedRanges = useDebouncedValue(previewChannelRanges, debounceMs);

  const payload = useMemo(() => toPayload(previewChannelRanges), [previewChannelRanges]);
  const debouncedPayload = useMemo(() => toPayload(debouncedRanges), [debouncedRanges]);

  const debouncedSignature = useMemo(() => {
    return Object.entries(debouncedRanges)
      .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }))
      .map(([channel, [min, max]]) => `${channel}:${min}-${max}`)
      .join("|");
  }, [debouncedRanges]);

  return { payload, debouncedPayload, debouncedSignature };
}
