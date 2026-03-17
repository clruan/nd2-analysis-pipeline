import { useMemo } from "react";
import { useAppStore } from "../state/useAppStore";
import { useDebouncedValue } from "./useDebouncedValue";

export function useThresholds() {
  const thresholds = useAppStore((state) => state.thresholds);
  const debounced = useDebouncedValue(thresholds, 250);
  return useMemo(() => ({ immediate: thresholds, debounced }), [thresholds, debounced]);
}
