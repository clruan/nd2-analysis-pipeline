import { useEffect, useMemo, useRef, useState } from "react";
import PlotlyChart from "./PlotlyChart";
import type { PlotHoverEvent, Shape, Annotations, PlotlyHTMLElement, DownloadImgopts } from "plotly.js";
import { Alert, Box, Button, CircularProgress, Stack, Typography } from "@mui/material";
import { apiClient } from "../api/client";
import { useAnalysisQuery, useDownloadMutation, useStatisticsQuery } from "../api/hooks";
import { useAppStore } from "../state/useAppStore";
import { useThresholds } from "../hooks/useThresholds";
import { thresholdsEqual, useThresholdMotion } from "../hooks/useThresholdMotion";
import type { IndividualImageRecord, MouseAverageRecord, StatisticsResponse } from "../api/types";
import { CHANNEL_METRICS, normalizeChannelDefinitions } from "../constants/metrics";
import DownloadIcon from "@mui/icons-material/FileDownloadOutlined";

type MetricDescriptor = {
  id: string;
  label: string;
  statsKey: string;
  valueAccessor: (record: MouseAverageRecord) => number | null;
  replicateAccessor: (record: IndividualImageRecord) => number | null;
};

type PairwiseComparison = {
  group1: string;
  group2: string;
  statistic: number;
  p_value: number;
  significance: string;
};

type StatisticalBlock = {
  comparison_mode: "all_vs_one" | "pairs" | "all_pairs";
  reference_group?: string | null;
  overall_test?: {
    statistic: number;
    p_value: number;
    significance: string;
  } | null;
  pairwise_comparisons?: PairwiseComparison[];
  note?: string | null;
};

interface SamplePoint {
  value: number;
  mouseId: string;
}

interface ReplicatePoint {
  id: string;
  value: number;
  label: string;
}

interface HoverState {
  metricId: string;
  subjectKey: string;
}

interface HoverTooltipState {
  metricId: string;
  left: number;
  top: number;
  title: string;
  details: string[];
  accentColor: string;
}

type PlotlyModule = typeof import("plotly.js");

const apiBase = (apiClient.defaults.baseURL ?? "").replace(/\/$/, "");

const defaultPalette = [
  "#2563eb",
  "#ea580c",
  "#16a34a",
  "#a855f7",
  "#f97316",
  "#d97706",
  "#0ea5e9",
  "#f43f5e",
  "#14b8a6",
  "#6366f1"
];

const baseLayout = {
  height: 380,
  autosize: true,
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  margin: { t: 48, r: 28, b: 68, l: 76 },
  font: { family: "Inter, sans-serif", color: "#0f172a", size: 13 },
  showlegend: false,
  hoverlabel: {
    font: { family: "Inter, sans-serif" },
    bgcolor: "#f8fafc",
    bordercolor: "#cbd5f5"
  },
  xaxis: {
    title: "",
    zeroline: false,
    showgrid: false,
    showline: true,
    linecolor: "#cbd5e1",
    tickfont: { color: "#0f172a" }
  },
  yaxis: {
    title: "",
    zeroline: false,
    gridcolor: "#e2e8f0",
    linecolor: "#cbd5e1",
    tickfont: { color: "#0f172a" }
  }
} as const;

const testTypeLabel: Record<string, string> = {
  anova_parametric: "Ordinary ANOVA",
  anova_non_parametric: "Non-parametric ANOVA",
  t_test: "Pairwise t-tests"
};

const formatPValue = (value: number) => {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  if (value === 0) {
    return "<1e-8";
  }
  if (value < 1e-3) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
};

const formatMetricValue = (value: number) => (Number.isFinite(value) ? value.toFixed(2) : "n/a");

const clampToRange = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function listGroups(mouseAverages: Array<{ Group: string }>): string[] {
  const unique = new Set<string>();
  mouseAverages.forEach((row) => unique.add(row.Group));
  return Array.from(unique);
}

function buildColorMap(base: Record<string, string>, groups: string[]): Record<string, string> {
  const palette: Record<string, string> = { ...base };
  groups.forEach((group, index) => {
    if (!palette[group]) {
      palette[group] = defaultPalette[index % defaultPalette.length];
    }
  });
  return palette;
}

function collectMetricValues(
  metric: MetricDescriptor,
  mouseAverages: MouseAverageRecord[],
  individualImages: IndividualImageRecord[]
): number[] {
  const subjectValues = mouseAverages
    .map((record) => metric.valueAccessor(record))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const replicateValues = individualImages
    .map((record) => metric.replicateAccessor(record))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return [...subjectValues, ...replicateValues];
}

function buildReplicateLookup(
  individualImages: IndividualImageRecord[],
  metrics: MetricDescriptor[]
): Record<string, Map<string, ReplicatePoint[]>> {
  const map: Record<string, Map<string, ReplicatePoint[]>> = {};
  metrics.forEach((metric) => {
    map[metric.id] = new Map();
  });
  individualImages.forEach((record) => {
    metrics.forEach((metric) => {
      const raw = metric.replicateAccessor(record);
      if (typeof raw !== "number" || !Number.isFinite(raw)) {
        return;
      }
      const key = `${record.group}|${record.mouse_id}`;
      const bucket = map[metric.id].get(key) ?? [];
      bucket.push({
        id: `${record.filename}|${record.replicate_index}`,
        value: raw,
        label: `${record.filename} (rep ${record.replicate_index})`
      });
      map[metric.id].set(key, bucket);
    });
  });
  return map;
}

const hashString = (value: string) => {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
};

const deterministicKeyOffset = (key: string, width: number) => {
  if (width <= 0) {
    return 0;
  }
  const fraction = (hashString(key) % 10000) / 9999;
  return (fraction - 0.5) * 2 * width;
};

type PlotShape = Partial<Shape>;
type PlotAnnotation = Partial<Annotations>;

const buildSignificanceLayers = (
  comparisons: PairwiseComparison[] | undefined,
  indexByGroup: Map<string, number>,
  baseMax: number
): { shapes: PlotShape[]; annotations: PlotAnnotation[]; maxY: number } => {
  const shapes: PlotShape[] = [];
  const annotations: PlotAnnotation[] = [];
  if (!comparisons || comparisons.length === 0) {
    const padding = baseMax > 0 ? baseMax * 0.15 : 1;
    return { shapes, annotations, maxY: baseMax + padding };
  }

  const safeBase = baseMax > 0 ? baseMax : 1;
  const band = safeBase * 0.18 + 0.8;
  let level = 0;

  comparisons.forEach((comparison) => {
    const start = indexByGroup.get(comparison.group1);
    const end = indexByGroup.get(comparison.group2);
    if (start === undefined || end === undefined) {
      return;
    }
    const left = Math.min(start, end);
    const right = Math.max(start, end);
    const height = safeBase + band * (level + 1);
    const arm = band * 0.45;

    shapes.push(
      {
        type: "line",
        xref: "x",
        yref: "y",
        x0: left,
        x1: right,
        y0: height,
        y1: height,
        line: { color: "#0f172a", width: 1 }
      },
      {
        type: "line",
        xref: "x",
        yref: "y",
        x0: left,
        x1: left,
        y0: height,
        y1: height - arm,
        line: { color: "#0f172a", width: 1 }
      },
      {
        type: "line",
        xref: "x",
        yref: "y",
        x0: right,
        x1: right,
        y0: height,
        y1: height - arm,
        line: { color: "#0f172a", width: 1 }
      }
    );

    annotations.push({
      xref: "x",
      yref: "y",
      x: (left + right) / 2,
      y: height + arm * 0.55,
      showarrow: false,
      text: comparison.significance,
      font: { size: 12, color: "#0f172a", family: "Inter, sans-serif" },
      captureevents: false
    });

    level += 1;
  });

  const maxY = safeBase + band * (level + 1.6);
  return { shapes, annotations, maxY };
};

export default function AnalysisBoard() {
  const {
    study,
    thresholds,
    statisticsEnabled,
    statisticsSettings,
    selectedMetric,
    setSelectedMetric,
    plotSettings,
    ratioDefinitions,
    channelDefinitions,
    thresholdControlHovered
  } = useAppStore((state) => ({
    study: state.study,
    thresholds: state.thresholds,
    statisticsEnabled: state.statisticsEnabled,
    statisticsSettings: state.statisticsSettings,
    selectedMetric: state.selectedMetric,
    setSelectedMetric: state.setSelectedMetric,
    plotSettings: state.plotSettings,
    ratioDefinitions: state.ratioDefinitions,
    channelDefinitions: state.channelDefinitions,
    thresholdControlHovered: state.thresholdControlHovered
  }));

  const { debounced } = useThresholds();
  const normalizedChannels = useMemo(() => normalizeChannelDefinitions(channelDefinitions), [channelDefinitions]);
  const channelLabelMap = useMemo(
    () =>
      normalizedChannels.reduce<Record<number, string>>((acc, definition) => {
        acc[definition.channel] = definition.label;
        return acc;
      }, {}),
    [normalizedChannels]
  );

  const metrics = useMemo<MetricDescriptor[]>(() => {
    const base = CHANNEL_METRICS.map((metric) => {
      const mouseKey = `Channel_${metric.channel}_area` as keyof MouseAverageRecord;
      const replicateKey = `channel_${metric.channel}_area` as keyof IndividualImageRecord;
      const channelLabel = channelLabelMap[metric.channel] ?? `Channel ${metric.channel}`;
      return {
        id: metric.id,
        label: `${channelLabel} Positive Signal (%)`,
        statsKey: `channel_${metric.channel}`,
        valueAccessor: (record: MouseAverageRecord) => {
          const raw = record[mouseKey];
          return typeof raw === "number" ? raw : null;
        },
        replicateAccessor: (record: IndividualImageRecord) => {
          const raw = record[replicateKey];
          return typeof raw === "number" ? raw : null;
        }
      };
    });
    const ratioMetrics = ratioDefinitions.map((ratio) => ({
      id: ratio.id,
      label: ratio.label,
      statsKey: ratio.id,
      valueAccessor: (record: MouseAverageRecord) => record.ratios?.[ratio.id] ?? null,
      replicateAccessor: (record: IndividualImageRecord) => record.ratios?.[ratio.id] ?? null
    }));
    return [...base, ...ratioMetrics];
  }, [channelLabelMap, ratioDefinitions]);

  const analysisQuery = useAnalysisQuery(study?.study_id ?? null, debounced);

  const statisticsActive =
    statisticsEnabled &&
    (statisticsSettings.comparisonMode !== "pairs" || statisticsSettings.comparisonPairs.length > 0);

  const statisticsQuery = useStatisticsQuery(study?.study_id ?? null, debounced, {
    enabled: statisticsActive,
    comparisonMode: statisticsSettings.comparisonMode,
    referenceGroup: statisticsSettings.referenceGroup ?? undefined,
    comparisonPairs:
      statisticsSettings.comparisonMode === "pairs"
        ? statisticsSettings.comparisonPairs.map((pair) => [pair[0], pair[1]])
        : undefined,
    testType: statisticsSettings.testType,
    significanceDisplay: statisticsSettings.significanceDisplay
  });

  const downloadMutation = useDownloadMutation(study?.study_id ?? null);

  const thresholdMotion = useThresholdMotion(analysisQuery.data);
  const displayedAnalysis = thresholdMotion.displayData ?? analysisQuery.data;
  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const [tooltipState, setTooltipState] = useState<HoverTooltipState | null>(null);
  const plotRefs = useRef<Record<string, PlotlyHTMLElement | null>>({});
  const hoverClearTimeoutRef = useRef<number | null>(null);
  const [exportingPlot, setExportingPlot] = useState<string | null>(null);
  const frozenStatisticsRef = useRef<StatisticsResponse | undefined>(statisticsQuery.data);
  const previousMotionActiveRef = useRef(false);

  useEffect(() => {
    return () => {
      if (hoverClearTimeoutRef.current !== null) {
        window.clearTimeout(hoverClearTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (thresholdMotion.active && !previousMotionActiveRef.current) {
      frozenStatisticsRef.current = statisticsQuery.data ?? frozenStatisticsRef.current;
    }
    if (!thresholdMotion.active && statisticsQuery.data) {
      frozenStatisticsRef.current = statisticsQuery.data;
    }
    previousMotionActiveRef.current = thresholdMotion.active;
  }, [thresholdMotion.active, statisticsQuery.data]);

  const mouseAverages = displayedAnalysis?.mouse_averages ?? [];
  const individualImages = displayedAnalysis?.individual_images ?? [];

  const groupNames = listGroups(mouseAverages);
  const colorMap = buildColorMap(plotSettings.palette, groupNames);
  const replicateLookup = buildReplicateLookup(individualImages, metrics);
  const jitterWidth = plotSettings.jitterEnabled ? plotSettings.jitterWidth : 0;
  const isUpdatingMetrics = analysisQuery.isFetching || statisticsQuery.isFetching;
  const targetThresholds = thresholdMotion.toData?.thresholds ?? analysisQuery.data?.thresholds ?? thresholds;
  const statisticsAreCurrent =
    Boolean(statisticsQuery.data) &&
    thresholdsEqual((statisticsQuery.data as StatisticsResponse).thresholds, targetThresholds);
  const displayedStatistics =
    thresholdMotion.active && (thresholdMotion.stage.subjectProgress < 0.92 || !statisticsAreCurrent)
      ? frozenStatisticsRef.current ?? statisticsQuery.data
      : statisticsQuery.data;

  const statisticsNote =
    statisticsEnabled && statisticsSettings.comparisonMode === "pairs" && statisticsSettings.comparisonPairs.length === 0;

  if (!study) {
    return (
      <Stack spacing={2} alignItems="center" justifyContent="center" sx={{ minHeight: "60vh" }}>
        <Typography variant="h6" color="text.secondary">
          Load a study to begin interactive analysis.
        </Typography>
      </Stack>
    );
  }

  if (analysisQuery.isLoading) {
    return (
      <Stack spacing={2} alignItems="center" justifyContent="center" sx={{ minHeight: "60vh" }}>
        <CircularProgress size={32} />
        <Typography variant="body2" color="text.secondary">
          Computing mouse-level metrics...
        </Typography>
      </Stack>
    );
  }

  if (analysisQuery.isError || !analysisQuery.data) {
    return <Alert severity="error">Unable to fetch analysis results.</Alert>;
  }

  const scheduleHoverClear = () => {
    if (hoverClearTimeoutRef.current !== null) {
      window.clearTimeout(hoverClearTimeoutRef.current);
    }
    hoverClearTimeoutRef.current = window.setTimeout(() => {
      setHoverState(null);
      setTooltipState(null);
      hoverClearTimeoutRef.current = null;
    }, 160);
  };

  const buildTooltipPosition = (metricId: string, event: MouseEvent) => {
    const graphDiv = plotRefs.current[metricId];
    if (!graphDiv) {
      return null;
    }
    const rect = graphDiv.getBoundingClientRect();
    const maxLeft = Math.max(16, rect.width - 232);
    const maxTop = Math.max(16, rect.height - 112);
    const preferredLeft = event.clientX - rect.left + 28;
    const preferredTop = event.clientY - rect.top - 64;
    const fallbackTop = event.clientY - rect.top + 28;
    return {
      left: clampToRange(preferredLeft, 16, maxLeft),
      top: clampToRange(preferredTop < 16 ? fallbackTop : preferredTop, 16, maxTop)
    };
  };

  const handleHover = (metricId: string, event: Readonly<PlotHoverEvent>) => {
    if (hoverClearTimeoutRef.current !== null) {
      window.clearTimeout(hoverClearTimeoutRef.current);
      hoverClearTimeoutRef.current = null;
    }
    const point = event.points?.[0];
    if (!point) {
      scheduleHoverClear();
      return;
    }
    const tooltipPosition = buildTooltipPosition(metricId, event.event);
    const traceName = String(point.data?.name ?? "");
    if (traceName === "Group mean" && tooltipPosition) {
      const raw = point.customdata as unknown;
      const group = Array.isArray(raw) && raw.length > 0 ? String(raw[0]) : String(point.x ?? "");
      const sd = Array.isArray(raw) && typeof raw[1] === "number" ? raw[1] : null;
      const count = Array.isArray(raw) && typeof raw[2] === "number" ? raw[2] : null;
      setHoverState(null);
      setTooltipState({
        metricId,
        ...tooltipPosition,
        title: group,
        details: [`Mean: ${formatMetricValue(Number(point.y))}`, `SD: ${formatMetricValue(sd ?? Number.NaN)}`, `n = ${count ?? "n/a"}`],
        accentColor: colorMap[group] ?? "#0f172a"
      });
      return;
    }
    if (traceName === "Subjects") {
      const raw = point.customdata as unknown;
      if (!Array.isArray(raw) || raw.length < 2) {
        scheduleHoverClear();
        return;
      }
      const group = String(raw[0]);
      const mouseId = String(raw[1]);
      const subjectKey = `${group}|${mouseId}`;
      const replicates = replicateLookup[metricId]?.get(subjectKey);
      if (replicates && replicates.length > 0) {
        setHoverState({ metricId, subjectKey });
      } else {
        setHoverState(null);
      }
      if (tooltipPosition) {
        setTooltipState({
          metricId,
          ...tooltipPosition,
          title: `${group} • ${mouseId}`,
          details: [
            `Subject value: ${formatMetricValue(Number(point.y))}`,
            `${replicates?.length ?? 0} replica${replicates && replicates.length === 1 ? "" : "s"}`
          ],
          accentColor: colorMap[group] ?? "#0f172a"
        });
      }
      return;
    }
    if (traceName === "Replicates") {
      const raw = point.customdata as unknown;
      if (Array.isArray(raw) && raw.length >= 4) {
        const subjectKey = String(raw[0]);
        const group = String(raw[1]);
        const mouseId = String(raw[2]);
        const label = String(raw[3]);
        setHoverState({ metricId, subjectKey });
        if (tooltipPosition) {
          setTooltipState({
            metricId,
            ...tooltipPosition,
            title: `${group} • ${mouseId}`,
            details: [label, `Replica value: ${formatMetricValue(Number(point.y))}`],
            accentColor: colorMap[group] ?? "#94a3b8"
          });
        }
        return;
      }
    }
    if (traceName === "Replica hover region") {
      const raw = point.customdata as unknown;
      const subjectKey = Array.isArray(raw) ? raw[0] : raw;
      if (typeof subjectKey === "string" && subjectKey) {
        setHoverState({ metricId, subjectKey });
        return;
      }
    }
    scheduleHoverClear();
  };

  const handleUnhover = () => {
    scheduleHoverClear();
  };

  const registerPlotHandle =
    (metricId: string) =>
    (_figure: unknown, graphDiv: PlotlyHTMLElement): void => {
      plotRefs.current[metricId] = graphDiv;
    };

  const handleExportFigure = async (metricId: string) => {
    const graphDiv = plotRefs.current[metricId];
    if (!graphDiv) {
      return;
    }
    setExportingPlot(metricId);
    try {
      const plotlyModule = await import("plotly.js-dist-min");
      const plotlyLib = (plotlyModule.default ?? plotlyModule) as PlotlyModule;
      const filenameParts = [
        (study.study_name || "microscopy-study").replace(/\s+/g, "_"),
        metricId,
        `${thresholds.channel_1}-${thresholds.channel_2}-${thresholds.channel_3}`
      ];
      const downloadOptions = {
        format: "png",
        filename: filenameParts.join("_"),
        width: 1280,
        height: 720,
        scale: 2
      } as DownloadImgopts;
      await plotlyLib.downloadImage(graphDiv, downloadOptions);
    } catch (error) {
      console.error("Failed to export plot", error);
    } finally {
      setExportingPlot((current) => (current === metricId ? null : current));
    }
  };


  return (
    <Stack spacing={3} sx={{ pb: 6 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h5" color="text.primary">
            {study.study_name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {groupNames.length} groups • thresholds: {(channelLabelMap[1] ?? "Ch1")} {thresholds.channel_1} | {(channelLabelMap[2] ?? "Ch2")} {thresholds.channel_2} | {(channelLabelMap[3] ?? "Ch3")} {thresholds.channel_3}
          </Typography>
          {isUpdatingMetrics && (
            <Typography variant="caption" color="text.secondary" display="block">
              Updating threshold metrics…
            </Typography>
          )}
        </Box>
        <Button
          variant="outlined"
          size="small"
          disabled={downloadMutation.isPending}
          onClick={async () => {
            try {
              const response = await downloadMutation.mutateAsync(thresholds);
              const url = `${apiBase}/studies/${study.study_id}/download-file?path=${encodeURIComponent(
                response.download_path
              )}`;
              window.open(url, "_blank", "noopener");
            } catch {
              /* handled via mutation error */
            }
          }}
        >
          {downloadMutation.isPending ? "Generating…" : "Download Excel"}
        </Button>
      </Stack>

      {plotSettings.title && (
        <Typography variant="subtitle1" color="text.primary">
          {plotSettings.title}
        </Typography>
      )}

      {downloadMutation.isError && (
        <Alert severity="error">Download failed: {String(downloadMutation.error ?? "unknown error")}</Alert>
      )}

      {statisticsNote && (
        <Alert severity="info">Select at least one comparison pair or change the mode to run statistics.</Alert>
      )}

      {statisticsEnabled && statisticsQuery.isError && (
        <Alert severity="error">Failed to compute statistics for the current thresholds.</Alert>
      )}

      {metrics.map((metric) => {
        const metricData = mouseAverages
          .map((row) => ({
            group: String(row.Group),
            mouseId: String(row.MouseID),
            value: metric.valueAccessor(row)
          }))
          .filter((entry) => typeof entry.value === "number" && Number.isFinite(entry.value as number));

        if (metricData.length === 0) {
          return null;
        }

        const grouped = groupNames.map((group) => {
          const samples: SamplePoint[] = metricData
            .filter((entry) => entry.group === group)
            .map((entry) => ({ value: entry.value as number, mouseId: entry.mouseId }));
          const values = samples.map((sample) => sample.value);
          const count = values.length;
          const mean = count > 0 ? values.reduce((acc, value) => acc + value, 0) / count : 0;
          const variance = count > 1 ? values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / (count - 1) : 0;
          const sd = count > 1 ? Math.sqrt(variance) : 0;
          return { group, samples, mean, sd, count };
        });

        const indexByGroup = new Map<string, number>();
        grouped.forEach((entry, index) => indexByGroup.set(entry.group, index));

        const groupIndices = grouped.map((_, index) => index);
        const groupMeans = grouped.map((entry) => entry.mean);
        const axisCandidates = [
          ...collectMetricValues(metric, mouseAverages, individualImages),
          ...(thresholdMotion.active && thresholdMotion.fromData
            ? collectMetricValues(metric, thresholdMotion.fromData.mouse_averages, thresholdMotion.fromData.individual_images)
            : []),
          ...(thresholdMotion.active && thresholdMotion.toData
            ? collectMetricValues(metric, thresholdMotion.toData.mouse_averages, thresholdMotion.toData.individual_images)
            : [])
        ].filter((value) => Number.isFinite(value));
        const baseMin = axisCandidates.length > 0 ? Math.min(...axisCandidates) : 0;
        const baseMax = axisCandidates.length > 0 ? Math.max(...axisCandidates) : 0;

        const statBlock = statisticsEnabled && displayedStatistics
          ? (displayedStatistics.statistics?.[metric.statsKey as keyof typeof displayedStatistics.statistics] as StatisticalBlock | undefined)
          : undefined;
        const frozenStatBlock = statisticsEnabled && frozenStatisticsRef.current
          ? (frozenStatisticsRef.current.statistics?.[
              metric.statsKey as keyof typeof frozenStatisticsRef.current.statistics
            ] as StatisticalBlock | undefined)
          : undefined;
        const latestStatBlock = statisticsEnabled && statisticsQuery.data
          ? (statisticsQuery.data.statistics?.[metric.statsKey as keyof typeof statisticsQuery.data.statistics] as StatisticalBlock | undefined)
          : undefined;

        const displayedLayers = buildSignificanceLayers(
          statBlock?.pairwise_comparisons,
          indexByGroup,
          baseMax
        );
        const frozenLayers = buildSignificanceLayers(
          frozenStatBlock?.pairwise_comparisons,
          indexByGroup,
          baseMax
        );
        const latestLayers = buildSignificanceLayers(
          latestStatBlock?.pairwise_comparisons,
          indexByGroup,
          baseMax
        );
        const { shapes, annotations } = displayedLayers;
        const significanceUpper = Math.max(displayedLayers.maxY, frozenLayers.maxY, latestLayers.maxY, baseMax || 1);
        const lowerCandidate = baseMin < 0 ? baseMin : 0;
        const ySpan = Math.max(significanceUpper - lowerCandidate, 1);
        const yLower = baseMin < 0 ? lowerCandidate - ySpan * 0.08 : 0;
        const yUpper = significanceUpper + ySpan * 0.06;

        const samplesX: number[] = [];
        const samplesY: number[] = [];
        const samplesText: string[] = [];
        const samplesCustom: [string, string, number][] = [];
        const subjectPointMap = new Map<string, { x: number; y: number; group: string; mouseId: string }>();
        const subjectJitterWidth = jitterWidth;

        grouped.forEach((entry, groupIndex) => {
          entry.samples
            .slice()
            .sort((first, second) => first.mouseId.localeCompare(second.mouseId, undefined, { numeric: true, sensitivity: "base" }))
            .forEach((sample) => {
            const subjectKey = `${entry.group}|${sample.mouseId}`;
            const x = groupIndex + deterministicKeyOffset(subjectKey, subjectJitterWidth);
            samplesX.push(x);
            samplesY.push(sample.value);
            samplesText.push(`${entry.group} • ${sample.mouseId}: ${sample.value.toFixed(2)}`);
            samplesCustom.push([entry.group, sample.mouseId, groupIndex]);
            subjectPointMap.set(subjectKey, {
              x,
              y: sample.value,
              group: entry.group,
              mouseId: sample.mouseId
            });
          });
        });

        const barTrace = {
          type: "bar" as const,
          x: groupIndices,
          y: groupMeans,
          marker: {
            color: grouped.map((entry) =>
              entry.group === statBlock?.reference_group ? "#1d4ed8" : colorMap[entry.group]
            ),
            opacity: 0.9,
            line: { color: "#0f172a", width: 0.6 }
          },
          width: 0.55,
          hovertemplate:
            "<b>%{customdata[0]}</b><br>Mean: %{y:.2f}<br>SD: %{customdata[1]:.2f}<br>n=%{customdata[2]}<extra></extra>",
          customdata: grouped.map((entry) => [entry.group, entry.sd, entry.count]),
          name: "Group mean",
          showlegend: false
        };

        const scatterTrace = {
          type: "scattergl" as const,
          mode: "markers" as const,
          x: samplesX,
          y: samplesY,
          text: samplesText,
          customdata: samplesCustom,
          marker: {
            color: samplesCustom.map(([group]) => colorMap[group]),
            size: 9,
            opacity: 0.78,
            line: { color: "#ffffff", width: 0.8 }
          },
          hovertemplate: "%{text}<extra></extra>",
          name: "Subjects",
          showlegend: false
        };

        const revealAllReplicates =
          thresholdControlHovered || (thresholdMotion.active && thresholdMotion.stage.replicaVisibility > 0.001);
        const showReplicaLinks = revealAllReplicates || (hoverState !== null && hoverState.metricId === metric.id);
        const revealedSubjectKeys = revealAllReplicates
          ? Array.from(subjectPointMap.keys())
          : hoverState && hoverState.metricId === metric.id
            ? [hoverState.subjectKey]
            : [];

        const replicateLinkX: Array<number | null> = [];
        const replicateLinkY: Array<number | null> = [];
        const repX: number[] = [];
        const repY: number[] = [];
        const repText: string[] = [];
        const repCustom: Array<[string, string, string, string]> = [];
        const replicaHoverX: number[] = [];
        const replicaHoverY: number[] = [];
        const replicaHoverCustom: string[] = [];

        revealedSubjectKeys.forEach((subjectKey) => {
          const subjectPoint = subjectPointMap.get(subjectKey);
          const replicates = replicateLookup[metric.id]?.get(subjectKey);
          if (!subjectPoint || !replicates || replicates.length === 0) {
            return;
          }
          const overlayWidth = Math.max(subjectJitterWidth * 0.55, 0.12);
          if (!revealAllReplicates) {
            replicaHoverX.push(subjectPoint.x);
            replicaHoverY.push(subjectPoint.y);
            replicaHoverCustom.push(subjectKey);
          }
          replicates.forEach((replicate) => {
            const repXValue = subjectPoint.x + deterministicKeyOffset(`${subjectKey}|${replicate.id}`, overlayWidth);
            repX.push(repXValue);
            repY.push(replicate.value);
            repText.push(
              `${subjectPoint.group} • ${subjectPoint.mouseId}\n${replicate.label}\n${replicate.value.toFixed(2)}`
            );
            repCustom.push([subjectKey, subjectPoint.group, subjectPoint.mouseId, replicate.label]);
            if (showReplicaLinks) {
              replicateLinkX.push(subjectPoint.x, repXValue, null);
              replicateLinkY.push(subjectPoint.y, replicate.value, null);
            }
            if (!revealAllReplicates) {
              replicaHoverX.push(repXValue, (subjectPoint.x + repXValue) / 2);
              replicaHoverY.push(replicate.value, (subjectPoint.y + replicate.value) / 2);
              replicaHoverCustom.push(subjectKey, subjectKey);
            }
          });
        });

        const linkOpacity = revealAllReplicates
          ? thresholdMotion.active
            ? 0.16 + thresholdMotion.stage.replicaVisibility * 0.26
            : 0.34
          : 0.28;

        const replicateLinkTrace =
          replicateLinkX.length > 0
            ? {
                type: "scattergl" as const,
                mode: "lines" as const,
                x: replicateLinkX,
                y: replicateLinkY,
                line: {
                  color: `rgba(15,23,42,${linkOpacity.toFixed(3)})`,
                  width: revealAllReplicates ? 1.1 : 1.25
                },
                hoverinfo: "skip" as const,
                name: "Replica links",
                showlegend: false
              }
            : null;

        const replicateTrace =
          repX.length > 0
            ? {
                type: "scattergl" as const,
                mode: "markers" as const,
                x: repX,
                y: repY,
                text: repText,
                customdata: repCustom,
                marker: {
                  color: "#94a3b8",
                  size: revealAllReplicates ? 5.2 + thresholdMotion.stage.replicaVisibility * 1.8 : 6.4,
                  opacity: revealAllReplicates ? 0.74 * thresholdMotion.stage.replicaVisibility : 0.88,
                  line: { color: "#ffffff", width: 0.6 }
                },
                hovertemplate: "%{text}<extra></extra>",
                name: "Replicates",
                showlegend: false
              }
            : null;

        const replicaHoverTrace =
          replicaHoverX.length > 0
            ? {
                type: "scattergl" as const,
                mode: "markers" as const,
                x: replicaHoverX,
                y: replicaHoverY,
                customdata: replicaHoverCustom,
                marker: {
                  size: 22,
                  color: "rgba(148,163,184,0.001)",
                  line: { width: 0 }
                },
                hovertemplate: "<extra></extra>",
                name: "Replica hover region",
                showlegend: false
              }
            : null;

        const data = [
          barTrace,
          replicateLinkTrace,
          replicateTrace,
          replicaHoverTrace,
          scatterTrace
        ].filter(Boolean);

        const overallTest = statBlock?.overall_test ?? null;
        const overallSummary =
          statisticsEnabled && overallTest
            ? `${testTypeLabel[statisticsSettings.testType] ?? statisticsSettings.testType}: ${overallTest.significance} (p=${formatPValue(
                overallTest.p_value
              )})`
            : null;

        return (
          <Box
            key={metric.id}
            role="button"
            tabIndex={0}
            onClick={() => setSelectedMetric(metric.id)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setSelectedMetric(metric.id);
              }
            }}
            sx={{
              borderRadius: 2,
              border: selectedMetric === metric.id ? "2px solid #2563eb" : "1px solid #d4d4d8",
              boxShadow: selectedMetric === metric.id ? "0 16px 32px rgba(37,99,235,0.12)" : "0 8px 24px rgba(15,23,42,0.06)",
              backgroundColor: "#ffffff",
              p: 2.5,
              outline: "none",
              cursor: "pointer",
              transition: "border 0.2s ease, box-shadow 0.2s ease",
              "&:hover": {
                boxShadow: "0 18px 34px rgba(15,23,42,0.12)"
              }
            }}
          >
            <Stack spacing={1.25}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
                <Typography variant="subtitle1" color="text.primary">
                  {metric.label}
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                  <Typography variant="caption" color="text.secondary">
                    n per group: {grouped.map((entry) => `${entry.group}=${entry.count}`).join(" • ") || "insufficient data"}
                  </Typography>
                  <Button
                    variant="text"
                    size="small"
                    startIcon={<DownloadIcon fontSize="small" />}
                    onClick={() => handleExportFigure(metric.id)}
                    disabled={exportingPlot === metric.id || !plotRefs.current[metric.id]}
                  >
                    {exportingPlot === metric.id ? "Exporting…" : "Export PNG"}
                  </Button>
                </Stack>
              </Stack>
              {statisticsEnabled && statisticsQuery.isLoading && (
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={14} />
                  <Typography variant="caption" color="text.secondary">
                    Updating statistics…
                  </Typography>
                </Stack>
              )}
              {overallSummary && (
                <Typography variant="caption" color="text.secondary">
                  {overallSummary}
                </Typography>
              )}
              <Box
                sx={{
                  position: "relative",
                  "& .hoverlayer": {
                    display: "none"
                  }
                }}
              >
                <PlotlyChart
                  data={data}
                  layout={{
                    ...baseLayout,
                    font: { ...baseLayout.font, size: plotSettings.fontSize },
                    margin: { ...baseLayout.margin, t: 40 },
                    dragmode: "pan",
                    uirevision: metric.id,
                    xaxis: {
                      ...baseLayout.xaxis,
                      tickvals: groupIndices,
                      ticktext: grouped.map((entry) => entry.group),
                      title: "Groups",
                      fixedrange: false,
                      range: [-0.6, groupIndices.length - 0.4]
                    },
                    yaxis: {
                      ...baseLayout.yaxis,
                      title: metric.label,
                      fixedrange: false,
                      range: [yLower, yUpper]
                    },
                    shapes,
                    annotations,
                    hovermode: "closest"
                  }}
                  config={{
                    responsive: true,
                    displaylogo: false,
                    displayModeBar: true,
                    scrollZoom: true,
                    doubleClick: "reset",
                    modeBarButtonsToRemove: ["lasso2d", "select2d"]
                  }}
                  style={{ width: "100%" }}
                  useResizeHandler
                  onInitialized={registerPlotHandle(metric.id)}
                  onUpdate={registerPlotHandle(metric.id)}
                  onHover={(event: PlotHoverEvent) => handleHover(metric.id, event)}
                  onUnhover={handleUnhover}
                />
                {tooltipState && tooltipState.metricId === metric.id && (
                  <Box
                    sx={{
                      position: "absolute",
                      left: tooltipState.left,
                      top: tooltipState.top,
                      zIndex: 2,
                      minWidth: 192,
                      maxWidth: 236,
                      px: 1.25,
                      py: 1,
                      borderRadius: 1.5,
                      border: "1px solid rgba(148,163,184,0.45)",
                      backgroundColor: "rgba(248,250,252,0.96)",
                      boxShadow: "0 14px 28px rgba(15,23,42,0.16)",
                      backdropFilter: "blur(10px)",
                      pointerEvents: "none",
                      borderLeft: `4px solid ${tooltipState.accentColor}`
                    }}
                  >
                    <Typography variant="caption" sx={{ display: "block", fontWeight: 700, color: "#0f172a" }}>
                      {tooltipState.title}
                    </Typography>
                    {tooltipState.details.map((detail) => (
                      <Typography
                        key={`${tooltipState.title}-${detail}`}
                        variant="caption"
                        sx={{ display: "block", color: "#475569", lineHeight: 1.45 }}
                      >
                        {detail}
                      </Typography>
                    ))}
                  </Box>
                )}
              </Box>
              {statBlock?.note && (
                <Typography variant="caption" color="text.secondary">
                  {statBlock.note}
                </Typography>
              )}
            </Stack>
          </Box>
        );
      })}
    </Stack>
  );
}
