import { useMemo, useRef, useState } from "react";
import PlotlyChart from "./PlotlyChart";
import type { PlotHoverEvent, Shape, Annotations, PlotlyHTMLElement, DownloadImgopts } from "plotly.js";
import { Alert, Box, Button, CircularProgress, Stack, Typography } from "@mui/material";
import { apiClient } from "../api/client";
import { useAnalysisQuery, useDownloadMutation, useStatisticsQuery, useInterpretationAssistant } from "../api/hooks";
import { useAppStore } from "../state/useAppStore";
import { useThresholds } from "../hooks/useThresholds";
import type { IndividualImageRecord, MouseAverageRecord } from "../api/types";
import { CHANNEL_METRICS } from "../constants/metrics";
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
  value: number;
  label: string;
}

interface HoverState {
  metricId: string;
  subjectKey: string;
  groupIndex: number;
}

interface MetricInsight {
  summary: string;
  bullets: string[];
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

const describeAssistantError = (error: unknown) => {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "Assistant unavailable. Try again soon.";
};

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
      bucket.push({ value: raw, label: `${record.filename} (rep ${record.replicate_index})` });
      map[metric.id].set(key, bucket);
    });
  });
  return map;
}

const deterministicOffset = (index: number, width: number) => {
  if (width <= 0) {
    return 0;
  }
  const phi = 0.61803398875;
  const fraction = ((index + 1) * phi) % 1;
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
    ratioDefinitions
  } = useAppStore((state) => ({
    study: state.study,
    thresholds: state.thresholds,
    statisticsEnabled: state.statisticsEnabled,
    statisticsSettings: state.statisticsSettings,
    selectedMetric: state.selectedMetric,
    setSelectedMetric: state.setSelectedMetric,
    plotSettings: state.plotSettings,
    ratioDefinitions: state.ratioDefinitions
  }));

  const { debounced } = useThresholds();
  const metrics = useMemo<MetricDescriptor[]>(() => {
    const base = CHANNEL_METRICS.map((metric) => {
      const mouseKey = `Channel_${metric.channel}_area` as keyof MouseAverageRecord;
      const replicateKey = `channel_${metric.channel}_area` as keyof IndividualImageRecord;
      return {
        id: metric.id,
        label: metric.label,
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
  }, [ratioDefinitions]);

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

  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const plotRefs = useRef<Record<string, PlotlyHTMLElement | null>>({});
  const [exportingPlot, setExportingPlot] = useState<string | null>(null);
  const [metricInsights, setMetricInsights] = useState<Record<string, MetricInsight>>({});
  const [pendingInsight, setPendingInsight] = useState<string | null>(null);
  const interpretationMutation = useInterpretationAssistant();

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

  const mouseAverages = analysisQuery.data.mouse_averages;
  const individualImages = analysisQuery.data.individual_images;

  const groupNames = listGroups(mouseAverages);
  const colorMap = buildColorMap(plotSettings.palette, groupNames);
  const replicateLookup = buildReplicateLookup(individualImages, metrics);
  const jitterWidth = plotSettings.jitterEnabled ? plotSettings.jitterWidth : 0;

  const statisticsNote =
    statisticsEnabled && statisticsSettings.comparisonMode === "pairs" && statisticsSettings.comparisonPairs.length === 0;

  const handleHover = (metricId: string, indexByGroup: Map<string, number>, event: Readonly<PlotHoverEvent>) => {
    const point = event.points?.[0];
    if (!point || point.data.name !== "Subjects") {
      setHoverState(null);
      return;
    }
    const raw = point.customdata as unknown;
    if (!Array.isArray(raw) || raw.length < 2) {
      setHoverState(null);
      return;
    }
    const group = String(raw[0]);
    const mouseId = String(raw[1]);
    const groupIndex = typeof raw[2] === "number" && Number.isFinite(raw[2]) ? (raw[2] as number) : indexByGroup.get(group) ?? 0;
    const subjectKey = `${group}|${mouseId}`;
    const replicates = replicateLookup[metricId]?.get(subjectKey);
    if (!replicates || replicates.length === 0) {
      setHoverState(null);
      return;
    }
    setHoverState({ metricId, subjectKey, groupIndex });
  };

  const handleUnhover = () => {
    setHoverState(null);
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
        (study.study_name || "nd2-study").replace(/\s+/g, "_"),
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

  const handleGenerateInsight = (
    metricId: string,
    metricLabel: string,
    groupedData: Array<{ group: string; mean: number; sd: number; count: number }>,
    statBlock?: StatisticalBlock
  ) => {
    const groupSummaries = groupedData.map((entry) => ({
      group: entry.group,
      mean: entry.mean,
      sd: Number.isFinite(entry.sd) ? entry.sd : null,
      count: entry.count
    }));
    if (!groupSummaries.length) {
      setMetricInsights((prev) => ({
        ...prev,
        [metricId]: {
          summary: "Need at least one subject per group to generate an interpretation.",
          bullets: []
        }
      }));
      return;
    }
    setPendingInsight(metricId);
    interpretationMutation.mutate(
      {
        metric_id: metricId,
        metric_label: metricLabel,
        group_summaries: groupSummaries,
        reference_group: statBlock?.reference_group ?? null,
        significance_notes:
          statBlock?.overall_test?.significance && Number.isFinite(statBlock.overall_test.p_value)
            ? `${statBlock.overall_test.significance} (p=${formatPValue(statBlock.overall_test.p_value)})`
            : statBlock?.note ?? null,
        thresholds
      },
      {
        onSuccess: (response) => {
          setMetricInsights((prev) => ({ ...prev, [metricId]: response }));
        },
        onError: (error) => {
          setMetricInsights((prev) => ({
            ...prev,
            [metricId]: {
              summary: "Assistant unavailable. Try again after refreshing.",
              bullets: [describeAssistantError(error)]
            }
          }));
        },
        onSettled: () => {
          setPendingInsight((current) => (current === metricId ? null : current));
        }
      }
    );
  };

  return (
    <Stack spacing={3} sx={{ pb: 6 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h5" color="text.primary">
            {study.study_name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {groupNames.length} groups • thresholds: Ch1 {thresholds.channel_1} | Ch2 {thresholds.channel_2} | Ch3 {thresholds.channel_3}
          </Typography>
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
        const sampleValues = grouped.flatMap((entry) => entry.samples.map((sample) => sample.value));
        const replicateBuckets = replicateLookup[metric.id];
        const replicateValues = replicateBuckets
          ? Array.from(replicateBuckets.values()).flatMap((entries) => entries.map((replicate) => replicate.value))
          : [];
        const axisCandidates = [...groupMeans, ...sampleValues, ...replicateValues].filter((value) =>
          Number.isFinite(value)
        );
        const baseMax = axisCandidates.length > 0 ? Math.max(...axisCandidates) : 0;

        const statBlock = statisticsEnabled && statisticsQuery.data
          ? (statisticsQuery.data.statistics?.[metric.statsKey as keyof typeof statisticsQuery.data.statistics] as StatisticalBlock | undefined)
          : undefined;

        const { shapes, annotations, maxY } = buildSignificanceLayers(
          statBlock?.pairwise_comparisons,
          indexByGroup,
          baseMax
        );

        const samplesX: number[] = [];
        const samplesY: number[] = [];
        const samplesText: string[] = [];
        const samplesCustom: [string, string, number][] = [];

        grouped.forEach((entry, groupIndex) => {
          entry.samples.forEach((sample, sampleIndex) => {
            const offset = deterministicOffset(sampleIndex, jitterWidth);
            samplesX.push(groupIndex + offset);
            samplesY.push(sample.value);
            samplesText.push(`${entry.group} • ${sample.mouseId}: ${sample.value.toFixed(2)}`);
            samplesCustom.push([entry.group, sample.mouseId, groupIndex]);
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
          type: "scatter" as const,
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

        let replicateTrace = null;
        if (hoverState && hoverState.metricId === metric.id) {
          const replicates = replicateLookup[metric.id]?.get(hoverState.subjectKey);
          if (replicates && replicates.length > 0) {
            const baseIndex = hoverState.groupIndex;
            const overlayWidth = jitterWidth > 0 ? jitterWidth * 0.6 : 0.12;
            const repX: number[] = [];
            const repY: number[] = [];
            const repText: string[] = [];
            replicates.forEach((replicate, replicateIndex) => {
              const offset = deterministicOffset(replicateIndex, overlayWidth);
              repX.push(baseIndex + offset);
              repY.push(replicate.value);
              repText.push(`${replicate.label}\n${replicate.value.toFixed(2)}`);
            });
            replicateTrace = {
              type: "scatter" as const,
              mode: "markers" as const,
              x: repX,
              y: repY,
              text: repText,
              marker: {
                color: "#64748b",
                size: 7,
                opacity: 0.85
              },
              hovertemplate: "%{text}<extra></extra>",
              name: "Replicates",
              showlegend: false
            };
          }
        }

        const data = replicateTrace ? [barTrace, scatterTrace, replicateTrace] : [barTrace, scatterTrace];

        const overallTest = statBlock?.overall_test ?? null;
        const overallSummary =
          statisticsEnabled && overallTest
            ? `${testTypeLabel[statisticsSettings.testType] ?? statisticsSettings.testType}: ${overallTest.significance} (p=${formatPValue(
                overallTest.p_value
              )})`
            : null;
        const insight = metricInsights[metric.id];

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
              <Stack direction="row" spacing={1} alignItems="center">
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() =>
                    handleGenerateInsight(
                      metric.id,
                      metric.label,
                      grouped.map((entry) => ({
                        group: entry.group,
                        mean: entry.mean,
                        sd: entry.sd,
                        count: entry.count
                      })),
                      statBlock
                    )
                  }
                  disabled={pendingInsight === metric.id && interpretationMutation.isPending}
                >
                  {pendingInsight === metric.id && interpretationMutation.isPending ? "Generating insight…" : "AI insight"}
                </Button>
                {insight && (
                  <Typography variant="caption" color="text.secondary">
                    Insight ready
                  </Typography>
                )}
              </Stack>
              {insight && (
                <Box
                  sx={{
                    borderLeft: "3px solid rgba(37,99,235,0.4)",
                    pl: 1.5,
                    py: 0.5,
                    backgroundColor: "rgba(37,99,235,0.02)"
                  }}
                >
                  <Typography variant="body2">{insight.summary}</Typography>
                  {insight.bullets.length > 0 && (
                    <Stack component="ul" spacing={0.25} sx={{ pl: 2, my: 0 }}>
                      {insight.bullets.map((bullet, index) => (
                        <Typography component="li" variant="caption" color="text.secondary" key={`${metric.id}-bullet-${index}`}>
                          {bullet}
                        </Typography>
                      ))}
                    </Stack>
                  )}
                </Box>
              )}
              <PlotlyChart
                data={data}
                layout={{
                  ...baseLayout,
                  font: { ...baseLayout.font, size: plotSettings.fontSize },
                  margin: { ...baseLayout.margin, t: 40 },
                  dragmode: "pan",
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
                    range: [0, maxY * 1.05]
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
                onHover={(event: PlotHoverEvent) => handleHover(metric.id, indexByGroup, event)}
                onUnhover={handleUnhover}
              />
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
