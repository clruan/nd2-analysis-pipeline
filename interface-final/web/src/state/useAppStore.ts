import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { LoadedStudy, RatioDefinition } from "../api/types";
import { DEFAULT_RATIO_DEFINITIONS } from "../constants/metrics";

type ThresholdMap = Record<string, number>;
type ChannelId = "channel_1" | "channel_2" | "channel_3";
type PanelId = "channel_1" | "channel_2" | "channel_3" | "composite";
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
  statisticsEnabled: boolean;
  statisticsSettings: StatisticsSettings;
  selectedMetric: string;
  plotSettings: PlotSettings;
  previewSamplesPerGroup: number;
  ratioDefinitions: RatioDefinition[];
  previewGroupOverrides: Record<string, number>;
  previewChannelRanges: Record<ChannelId, ChannelRangeTuple>;
  previewPanelOrder: PanelId[];
  setStudy: (study: LoadedStudy | null) => void;
  setThreshold: (channel: string, value: number) => void;
  setThresholds: (values: ThresholdMap) => void;
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
  setPreviewChannelRange: (channel: ChannelId, range: ChannelRangeTuple) => void;
  resetPreviewChannelRanges: () => void;
  setPreviewPanelOrder: (order: PanelId[]) => void;
  resetPreviewPanelOrder: () => void;
  setRatioDefinitions: (ratios: RatioDefinition[]) => void;
  setPreviewGroupOverride: (group: string, value: number | null) => void;
  resetPreviewGroupOverrides: () => void;
  updateStudy: (update: Partial<LoadedStudy>) => void;
}

const defaultThresholds: ThresholdMap = {
  channel_1: 2500,
  channel_2: 2500,
  channel_3: 300
};

const defaultSelectedMetric = "channel_1_area";
const createDefaultChannelRanges = (): Record<ChannelId, ChannelRangeTuple> => ({
  channel_1: [100, 2200],
  channel_2: [150, 2200],
  channel_3: [50, 2200]
});
const defaultPanelOrder: PanelId[] = ["channel_1", "channel_2", "channel_3", "composite"];

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

export const useAppStore = create<AppState>()(
  devtools((set) => ({
    study: null,
    thresholds: defaultThresholds,
    statisticsEnabled: false,
    statisticsSettings: createDefaultStatisticsSettings(),
    selectedMetric: defaultSelectedMetric,
    plotSettings: defaultPlotSettings(),
    previewSamplesPerGroup: 1,
    ratioDefinitions: DEFAULT_RATIO_DEFINITIONS,
    previewGroupOverrides: {},
    previewChannelRanges: createDefaultChannelRanges(),
    previewPanelOrder: [...defaultPanelOrder],
    setStudy: (study) =>
      set(() => ({
        study,
        statisticsEnabled: false,
        statisticsSettings: createDefaultStatisticsSettings(),
        selectedMetric: defaultSelectedMetric,
        plotSettings: defaultPlotSettings(),
        previewSamplesPerGroup: 1,
        ratioDefinitions: study?.ratio_definitions ?? DEFAULT_RATIO_DEFINITIONS,
        previewGroupOverrides: {},
        previewChannelRanges: createDefaultChannelRanges(),
        previewPanelOrder: [...defaultPanelOrder]
      })),
    setThreshold: (channel, value) =>
      set((state) => ({ thresholds: { ...state.thresholds, [channel]: value } })),
    setThresholds: (values) => set({ thresholds: { ...defaultThresholds, ...values } }),
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
        previewSamplesPerGroup: Math.max(1, Math.min(6, Math.round(value)))
      })),
    setPreviewChannelRange: (channel, range) =>
      set((state) => {
        const [rawMin, rawMax] = range;
        const min = Math.max(0, Math.min(4095, Math.min(rawMin, rawMax)));
        const max = Math.max(0, Math.min(4095, Math.max(rawMin, rawMax)));
        const adjusted: ChannelRangeTuple = min === max ? [min, min + 1] : [min, max];
        return {
          previewChannelRanges: {
            ...state.previewChannelRanges,
            [channel]: adjusted
          }
        };
      }),
    resetPreviewChannelRanges: () =>
      set(() => ({
        previewChannelRanges: createDefaultChannelRanges()
      })),
    setPreviewPanelOrder: (order) =>
      set(() => {
        const normalized = (order ?? []).filter(
          (panel, index, array): panel is PanelId =>
            ["channel_1", "channel_2", "channel_3", "composite"].includes(panel) &&
            array.indexOf(panel) === index
        );
        return {
          previewPanelOrder: normalized.length ? normalized : [...defaultPanelOrder]
        };
      }),
    resetPreviewPanelOrder: () =>
      set(() => ({
        previewPanelOrder: [...defaultPanelOrder]
      })),
    setRatioDefinitions: (ratios) =>
      set(() => ({
        ratioDefinitions: ratios.length ? ratios : DEFAULT_RATIO_DEFINITIONS
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
