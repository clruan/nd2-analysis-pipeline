import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { ChannelDefinition, LoadedStudy, RatioDefinition } from "../api/types";
import {
  DEFAULT_CHANNEL_DEFINITIONS,
  DEFAULT_RATIO_DEFINITIONS,
  normalizeChannelDefinitions
} from "../constants/metrics";

type ThresholdMap = Record<string, number>;
type PanelId = string;
type ChannelRangeTuple = [number, number];

type ComparisonMode = "all_vs_one" | "pairs" | "all_pairs";
type TestType = "anova_parametric" | "anova_non_parametric" | "t_test";
type SignificanceDisplay = "stars" | "p_values";
type ComparisonPair = [string, string];

interface StatisticsSettings {
  comparisonMode: ComparisonMode;
  referenceGroup: string | null;
  comparisonPairs: ComparisonPair[];
  testType: TestType;
  significanceDisplay: SignificanceDisplay;
}

interface PlotSettings {
  title: string;
  fontSize: number;
  palette: Record<string, string>;
  jitterEnabled: boolean;
  jitterWidth: number;
}

interface AppState {
  study: LoadedStudy | null;
  thresholds: ThresholdMap;
  thresholdControlHovered: boolean;
  statisticsEnabled: boolean;
  statisticsSettings: StatisticsSettings;
  selectedMetric: string;
  plotSettings: PlotSettings;
  previewSamplesPerGroup: number;
  ratioDefinitions: RatioDefinition[];
  channelDefinitions: ChannelDefinition[];
  previewGroupOverrides: Record<string, number>;
  previewChannelRanges: Record<string, ChannelRangeTuple>;
  previewPanelOrder: PanelId[];
  previewCompositeChannels: number[];
  previewScaleBarEnabled: boolean;
  previewScaleBarLengthUm: number;
  previewScaleBarFontSize: number;
  setStudy: (study: LoadedStudy | null) => void;
  setThreshold: (channel: string, value: number) => void;
  setThresholds: (values: ThresholdMap) => void;
  setThresholdControlHovered: (value: boolean) => void;
  setStatisticsEnabled: (value: boolean) => void;
  setSelectedMetric: (metricId: string) => void;
  setComparisonMode: (mode: ComparisonMode) => void;
  setReferenceGroup: (group: string | null) => void;
  addComparisonPair: (pair: ComparisonPair) => void;
  removeComparisonPair: (pair: ComparisonPair) => void;
  clearComparisonPairs: () => void;
  setSignificanceDisplay: (display: SignificanceDisplay) => void;
  setTestType: (testType: TestType) => void;
  resetStatisticsSettings: () => void;
  setPlotTitle: (title: string) => void;
  setPlotFontSize: (size: number) => void;
  setPaletteColor: (group: string, color: string) => void;
  setPalette: (palette: Record<string, string>) => void;
  setJitterEnabled: (enabled: boolean) => void;
  setJitterWidth: (width: number) => void;
  setPreviewSamplesPerGroup: (value: number) => void;
  setPreviewChannelRange: (channel: string, range: ChannelRangeTuple) => void;
  resetPreviewChannelRanges: () => void;
  setPreviewPanelOrder: (order: PanelId[]) => void;
  resetPreviewPanelOrder: () => void;
  setPreviewCompositeChannels: (channels: number[]) => void;
  resetPreviewCompositeChannels: () => void;
  setPreviewScaleBarEnabled: (value: boolean) => void;
  setPreviewScaleBarLengthUm: (value: number) => void;
  setPreviewScaleBarFontSize: (value: number) => void;
  setRatioDefinitions: (ratios: RatioDefinition[]) => void;
  setChannelDefinitions: (channels: ChannelDefinition[]) => void;
  setPreviewGroupOverride: (group: string, value: number | null) => void;
  resetPreviewGroupOverrides: () => void;
  updateStudy: (update: Partial<LoadedStudy>) => void;
}

const defaultSelectedMetric = "channel_1_area";

const createDefaultStatisticsSettings = (): StatisticsSettings => ({
  comparisonMode: "all_vs_one",
  referenceGroup: null,
  comparisonPairs: [],
  testType: "anova_parametric",
  significanceDisplay: "stars"
});

const defaultPlotSettings = (): PlotSettings => ({
  title: "",
  fontSize: 13,
  palette: {},
  jitterEnabled: false,
  jitterWidth: 0.12
});

const normalizePair = (pair: ComparisonPair): ComparisonPair => {
  const [a, b] = pair;
  if (a <= b) {
    return [a, b];
  }
  return [b, a];
};

const channelKey = (channel: number) => `channel_${channel}`;

const studyChannelIds = (study?: LoadedStudy | null) =>
  normalizeChannelDefinitions(study?.channel_definitions).map((entry) => entry.channel);

const defaultThresholdForChannel = (channel: number, limit: number) => {
  const fallbackMap: Record<number, number> = {
    1: 2500,
    2: 2500,
    3: 300
  };
  const fallback = fallbackMap[channel] ?? Math.round(limit * 0.5);
  return Math.max(0, Math.min(limit, fallback));
};

const createDefaultThresholds = (study?: LoadedStudy | null): ThresholdMap => {
  const ids = studyChannelIds(study);
  const definitions = normalizeChannelDefinitions(study?.channel_definitions, ids);
  return definitions.reduce<ThresholdMap>((acc, definition) => {
    const key = channelKey(definition.channel);
    const limit = Math.max(1, study?.channel_limits?.[key] ?? study?.max_threshold ?? 4095);
    acc[key] = defaultThresholdForChannel(definition.channel, limit);
    return acc;
  }, {});
};

const createDefaultChannelRanges = (study?: LoadedStudy | null): Record<string, ChannelRangeTuple> => {
  const ids = studyChannelIds(study);
  const definitions = normalizeChannelDefinitions(study?.channel_definitions, ids);
  return definitions.reduce<Record<string, ChannelRangeTuple>>((acc, definition) => {
    const key = channelKey(definition.channel);
    const limit = Math.max(1, study?.channel_limits?.[key] ?? study?.max_threshold ?? 4095);
    acc[key] = [0, limit];
    return acc;
  }, {});
};

const createDefaultPanelOrder = (study?: LoadedStudy | null): PanelId[] => {
  const ids = studyChannelIds(study);
  return [...ids.map((channel) => channelKey(channel)), "composite"];
};

const createDefaultCompositeChannels = (study?: LoadedStudy | null): number[] => {
  const ids = studyChannelIds(study);
  return ids.length ? ids : [1, 2, 3];
};

const defaultPreviewScaleBarLengthUm = 50;
const defaultPreviewScaleBarFontSize = 12;

const clampToLimit = (value: number, limit: number) => Math.max(0, Math.min(limit, Math.round(value)));

const normalizePreviewPanelOrder = (order: PanelId[], study?: LoadedStudy | null): PanelId[] => {
  const defaultOrder = createDefaultPanelOrder(study);
  const allowed = new Set(defaultOrder);
  const normalized = (order ?? []).filter((panel, index, array) => allowed.has(panel) && array.indexOf(panel) === index);
  return normalized.length ? normalized : defaultOrder;
};

export const useAppStore = create<AppState>()(
  devtools((set) => ({
    study: null,
    thresholds: createDefaultThresholds(),
    thresholdControlHovered: false,
    statisticsEnabled: false,
    statisticsSettings: createDefaultStatisticsSettings(),
    selectedMetric: defaultSelectedMetric,
    plotSettings: defaultPlotSettings(),
    previewSamplesPerGroup: 4,
    ratioDefinitions: DEFAULT_RATIO_DEFINITIONS,
    channelDefinitions: DEFAULT_CHANNEL_DEFINITIONS,
    previewGroupOverrides: {},
    previewChannelRanges: createDefaultChannelRanges(),
    previewPanelOrder: createDefaultPanelOrder(),
    previewCompositeChannels: createDefaultCompositeChannels(),
    previewScaleBarEnabled: true,
    previewScaleBarLengthUm: defaultPreviewScaleBarLengthUm,
    previewScaleBarFontSize: defaultPreviewScaleBarFontSize,
    setStudy: (study) =>
      set(() => {
        const normalizedChannels = normalizeChannelDefinitions(study?.channel_definitions);
        return {
          study,
          thresholds: createDefaultThresholds(study),
          thresholdControlHovered: false,
          statisticsEnabled: false,
          statisticsSettings: createDefaultStatisticsSettings(),
          selectedMetric: defaultSelectedMetric,
          plotSettings: defaultPlotSettings(),
          previewSamplesPerGroup: 4,
          ratioDefinitions: study?.ratio_definitions ?? DEFAULT_RATIO_DEFINITIONS,
          channelDefinitions: normalizedChannels,
          previewGroupOverrides: {},
          previewChannelRanges: createDefaultChannelRanges(study),
          previewPanelOrder: createDefaultPanelOrder(study),
          previewCompositeChannels: createDefaultCompositeChannels(study),
          previewScaleBarEnabled: true,
          previewScaleBarLengthUm: defaultPreviewScaleBarLengthUm,
          previewScaleBarFontSize: defaultPreviewScaleBarFontSize
        };
      }),
    setThreshold: (channel, value) =>
      set((state) => {
        const limit = Math.max(1, state.study?.channel_limits?.[channel] ?? state.study?.max_threshold ?? value ?? 0);
        const normalized = clampToLimit(value, limit);
        if (state.thresholds[channel] === normalized) {
          return state;
        }
        return { thresholds: { ...state.thresholds, [channel]: normalized } };
      }),
    setThresholds: (values) =>
      set((state) => {
        const defaults = createDefaultThresholds(state.study);
        const merged: ThresholdMap = { ...defaults };
        Object.entries(values).forEach(([channel, value]) => {
          const limit = Math.max(1, state.study?.channel_limits?.[channel] ?? state.study?.max_threshold ?? value ?? 0);
          merged[channel] = clampToLimit(value, limit);
        });
        return { thresholds: merged };
      }),
    setThresholdControlHovered: (value) => set({ thresholdControlHovered: value }),
    setStatisticsEnabled: (value) => set({ statisticsEnabled: value }),
    setSelectedMetric: (metricId) => set({ selectedMetric: metricId }),
    setComparisonMode: (mode) =>
      set((state) => {
        const nextSettings: StatisticsSettings = {
          ...state.statisticsSettings,
          comparisonMode: mode
        };
        if (mode !== "pairs") {
          nextSettings.comparisonPairs = [];
        }
        if (mode !== "all_vs_one") {
          nextSettings.referenceGroup = null;
        }
        return { statisticsSettings: nextSettings };
      }),
    setReferenceGroup: (group) =>
      set((state) => ({
        statisticsSettings: {
          ...state.statisticsSettings,
          referenceGroup: group
        }
      })),
    addComparisonPair: (pair) =>
      set((state) => {
        const normalized = normalizePair(pair);
        const exists = state.statisticsSettings.comparisonPairs.some((entry) => {
          const [a, b] = normalizePair(entry);
          return a === normalized[0] && b === normalized[1];
        });
        if (exists) {
          return {};
        }
        return {
          statisticsSettings: {
            ...state.statisticsSettings,
            comparisonPairs: [...state.statisticsSettings.comparisonPairs, normalized]
          }
        };
      }),
    removeComparisonPair: (pair) =>
      set((state) => {
        const normalized = normalizePair(pair);
        return {
          statisticsSettings: {
            ...state.statisticsSettings,
            comparisonPairs: state.statisticsSettings.comparisonPairs.filter((entry) => {
              const [a, b] = normalizePair(entry);
              return !(a === normalized[0] && b === normalized[1]);
            })
          }
        };
      }),
    clearComparisonPairs: () =>
      set((state) => ({
        statisticsSettings: {
          ...state.statisticsSettings,
          comparisonPairs: []
        }
      })),
    setSignificanceDisplay: (display) =>
      set((state) => ({
        statisticsSettings: {
          ...state.statisticsSettings,
          significanceDisplay: display
        }
      })),
    setTestType: (testType) =>
      set((state) => ({
        statisticsSettings: {
          ...state.statisticsSettings,
          testType
        }
      })),
    resetStatisticsSettings: () =>
      set(() => ({
        statisticsSettings: createDefaultStatisticsSettings(),
        statisticsEnabled: false
      })),
    setPlotTitle: (title) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          title
        }
      })),
    setPlotFontSize: (size) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          fontSize: Math.max(10, Math.min(24, size))
        }
      })),
    setPaletteColor: (group, color) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          palette: {
            ...state.plotSettings.palette,
            [group]: color
          }
        }
      })),
    setPalette: (palette) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          palette
        }
      })),
    setJitterEnabled: (enabled) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          jitterEnabled: enabled
        }
      })),
    setJitterWidth: (width) =>
      set((state) => ({
        plotSettings: {
          ...state.plotSettings,
          jitterWidth: Math.max(0, Math.min(0.4, width))
        }
      })),
    setPreviewSamplesPerGroup: (value) =>
      set(() => ({
        previewSamplesPerGroup: Math.max(1, Math.min(20, Math.round(value)))
      })),
    setPreviewChannelRange: (channel, range) =>
      set((state) => {
        const [rawMin, rawMax] = range;
        const limit = Math.max(1, state.study?.channel_limits?.[channel] ?? state.study?.max_threshold ?? Math.max(rawMin, rawMax));
        const min = clampToLimit(Math.min(rawMin, rawMax), limit);
        const max = clampToLimit(Math.max(rawMin, rawMax), limit);
        const adjusted: ChannelRangeTuple = min === max ? [min, Math.min(limit, min + 1)] : [min, max];
        const current = state.previewChannelRanges[channel];
        if (current && current[0] === adjusted[0] && current[1] === adjusted[1]) {
          return state;
        }
        return {
          previewChannelRanges: {
            ...state.previewChannelRanges,
            [channel]: adjusted
          }
        };
      }),
    resetPreviewChannelRanges: () =>
      set((state) => ({
        previewChannelRanges: createDefaultChannelRanges(state.study)
      })),
    setPreviewPanelOrder: (order) =>
      set((state) => ({
        previewPanelOrder: normalizePreviewPanelOrder(order, state.study)
      })),
    resetPreviewPanelOrder: () =>
      set((state) => ({
        previewPanelOrder: createDefaultPanelOrder(state.study)
      })),
    setPreviewCompositeChannels: (channels) =>
      set((state) => {
        const available = new Set(studyChannelIds(state.study));
        const normalized = Array.from(new Set(channels.map((channel) => Number(channel)).filter((channel) => available.has(channel)))).sort(
          (a, b) => a - b
        );
        return {
          previewCompositeChannels: normalized.length ? normalized : createDefaultCompositeChannels(state.study)
        };
      }),
    resetPreviewCompositeChannels: () =>
      set((state) => ({
        previewCompositeChannels: createDefaultCompositeChannels(state.study)
      })),
    setPreviewScaleBarEnabled: (value) =>
      set(() => ({
        previewScaleBarEnabled: value
      })),
    setPreviewScaleBarLengthUm: (value) =>
      set(() => ({
        previewScaleBarLengthUm: Math.max(5, Math.min(500, Math.round(value)))
      })),
    setPreviewScaleBarFontSize: (value) =>
      set(() => ({
        previewScaleBarFontSize: Math.max(6, Math.min(32, Math.round(value)))
      })),
    setRatioDefinitions: (ratios) =>
      set(() => ({
        ratioDefinitions: ratios.length ? ratios : DEFAULT_RATIO_DEFINITIONS
      })),
    setChannelDefinitions: (channels) =>
      set((state) => ({
        channelDefinitions: normalizeChannelDefinitions(channels, studyChannelIds(state.study))
      })),
    setPreviewGroupOverride: (group, value) =>
      set((state) => {
        const next = { ...state.previewGroupOverrides };
        if (value === null || Number.isNaN(value)) {
          delete next[group];
        } else {
          next[group] = Math.max(1, Math.min(6, Math.round(value)));
        }
        return { previewGroupOverrides: next };
      }),
    resetPreviewGroupOverrides: () =>
      set(() => ({
        previewGroupOverrides: {}
      })),
    updateStudy: (update) =>
      set((state) => ({
        study: state.study ? { ...state.study, ...update } : state.study
      }))
  }))
);
