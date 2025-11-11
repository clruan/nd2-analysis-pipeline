import { Fragment, useMemo, useState } from "react";
import { Alert, Box, Button, CircularProgress, IconButton, Slider, Stack, TextField, Tooltip, Typography } from "@mui/material";
import { useAppStore } from "../state/useAppStore";
import { useThresholds } from "../hooks/useThresholds";
import { usePreviewQuery, useClearPreviewsMutation } from "../api/hooks";
import type { PreviewImage } from "../api/types";
import { apiClient } from "../api/client";
import DownloadIcon from "@mui/icons-material/FileDownloadOutlined";
import RefreshIcon from "@mui/icons-material/RefreshOutlined";
import { CHANNEL_METRICS } from "../constants/metrics";

type Variant = PreviewImage["variant"];

interface PreviewColumn {
  key: string;
  group: string;
  subjectId: string;
  filename: string;
  metrics: Map<string, Map<Variant, PreviewImage[]>>;
}

const variantOrder: Variant[] = ["raw", "mask", "overlay"];

const variantLabels: Record<Variant, string> = {
  raw: "Raw",
  mask: "Mask",
  overlay: "Overlay"
};

const apiBase = (apiClient.defaults.baseURL ?? "").replace(/\/$/, "");
const imageHeight = 112;
const imageWidth = 130;

const previewKey = (image: PreviewImage) => `${image.variant}:${image.channel ?? "all"}`;

export default function PreviewPane() {
  const {
    study,
    statisticsSettings,
    selectedMetric,
    previewSamplesPerGroup,
    setPreviewSamplesPerGroup,
    ratioDefinitions,
    previewGroupOverrides,
    setPreviewGroupOverride,
    resetPreviewGroupOverrides
  } = useAppStore();
  const { debounced } = useThresholds();
  const metricsCatalog = useMemo(() => {
    const base = CHANNEL_METRICS.map((metric) => ({
      id: metric.id,
      title: metric.label
    }));
    const ratios = ratioDefinitions.map((ratio) => ({
      id: ratio.id,
      title: ratio.label
    }));
    return [...base, ...ratios];
  }, [ratioDefinitions]);
  const metricsForPreview = useMemo(() => metricsCatalog.map((metric) => metric.id), [metricsCatalog]);
  const metricLabels = useMemo(
    () =>
      metricsCatalog.reduce<Record<string, string>>((acc, metric) => {
        acc[metric.id] = metric.title;
        return acc;
      }, {}),
    [metricsCatalog]
  );
  const [showOverrides, setShowOverrides] = useState(false);

  const previewQuery = usePreviewQuery(
    study?.study_id ?? null,
    debounced,
    metricsForPreview,
    previewSamplesPerGroup,
    undefined,
    previewGroupOverrides
  );
  const clearCacheMutation = useClearPreviewsMutation(study?.study_id ?? null);
  const refreshMutation = useClearPreviewsMutation(study?.study_id ?? null);
  const thresholdKey = `${debounced.channel_1}-${debounced.channel_2}-${debounced.channel_3}`;
  const handleRefreshPreviews = () => {
    if (!study) return;
    refreshMutation.mutate(
      { scope: "thresholds", threshold_key: thresholdKey },
      {
        onSuccess: () => {
          void previewQuery.refetch();
        }
      }
    );
  };
  const handleFlushCache = () => {
    if (!study) return;
    clearCacheMutation.mutate(
      { scope: "all" },
      {
        onSuccess: () => {
          void previewQuery.refetch();
        }
      }
    );
  };

  const groupedPreviews = useMemo<PreviewColumn[]>(() => {
    if (!previewQuery.data) return [];
    const byGroup = new Map<string, PreviewColumn>();

    previewQuery.data.images.forEach((image) => {
      const key = `${image.group}|${image.subject_id}`;
      const existing = byGroup.get(key);
      if (existing) {
        const metricBucket = existing.metrics.get(image.metric) ?? new Map<Variant, PreviewImage[]>();
        const bucket = [...(metricBucket.get(image.variant) ?? []), image];
        bucket.sort((a, b) => (a.channel ?? 99) - (b.channel ?? 99));
        metricBucket.set(image.variant, bucket);
        existing.metrics.set(image.metric, metricBucket);
        return;
      }

      const metricMap = new Map<string, Map<Variant, PreviewImage[]>>();
      const initialBucket = new Map<Variant, PreviewImage[]>();
      initialBucket.set(image.variant, [image]);
      metricMap.set(image.metric, initialBucket);

      byGroup.set(key, {
        key,
        group: image.group,
        subjectId: image.subject_id,
        filename: image.filename,
        metrics: metricMap
      });
    });

    return Array.from(byGroup.values()).sort((a, b) => a.group.localeCompare(b.group));
  }, [previewQuery.data]);

  const activeGroups = useMemo(() => {
    if (!study) return null;
    if (statisticsSettings.comparisonMode === "pairs" && statisticsSettings.comparisonPairs.length > 0) {
      return new Set(statisticsSettings.comparisonPairs.flat());
    }
    if (statisticsSettings.comparisonMode === "all_vs_one" && statisticsSettings.referenceGroup) {
      const groups = new Set<string>();
      groups.add(statisticsSettings.referenceGroup);
      study.groups.forEach((group) => groups.add(group));
      return groups;
    }
    return null;
  }, [statisticsSettings, study]);

  const metricsWithData = useMemo(() => {
    return metricsCatalog.filter((metric) =>
      groupedPreviews.some((column) => {
        const bucket = column.metrics.get(metric.id);
        if (!bucket) return false;
        return variantOrder.some((variant) => (bucket.get(variant)?.length ?? 0) > 0);
      })
    );
  }, [groupedPreviews, metricsCatalog]);

  if (!study) {
    return (
      <Stack spacing={2} alignItems="center" justifyContent="center" sx={{ minHeight: "60vh" }}>
        <Typography variant="body2" color="text.secondary">
          Load a study to view live previews.
        </Typography>
      </Stack>
    );
  }

  if (previewQuery.isError) {
    return <Alert severity="error">Unable to load preview images.</Alert>;
  }

  const isRefreshing = previewQuery.isFetching && Boolean(previewQuery.data);
  const nd2Unavailable = Boolean(previewQuery.data && previewQuery.data.nd2_available === false);
  const nd2Source = previewQuery.data?.nd2_source ?? "";
  if (!previewQuery.data || groupedPreviews.length === 0) {
    return (
      <Stack spacing={1.5} px={1} py={2}>
        <Typography variant="h6">Live Previews</Typography>
        <Typography variant="body2" color="text.secondary">
          Adjust thresholds to generate preview overlays. Use the selector to request additional subjects per group.
        </Typography>
      </Stack>
    );
  }

  const token = previewQuery.data.generated_at;
  const metricLabel = metricLabels[selectedMetric] ?? "Selected metric";
  const referenceGroup = statisticsSettings.referenceGroup;

  return (
    <Stack spacing={1.5} sx={{ minHeight: "100%" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 2, flexWrap: "wrap" }}>
        <Box>
          <Typography variant="h6">Live Previews</Typography>
          <Typography variant="caption" color="text.secondary">
            Generated {new Date(token).toLocaleTimeString()}
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block">
            Focus: {metricLabel}
          </Typography>
          {referenceGroup && statisticsSettings.comparisonMode === "all_vs_one" && (
            <Typography variant="caption" color="text.secondary" display="block">
              Reference group: {referenceGroup}
            </Typography>
          )}
          {isRefreshing && (
            <Stack direction="row" spacing={0.5} alignItems="center" mt={0.5}>
              <CircularProgress size={10} thickness={6} />
              <Typography variant="caption" color="text.secondary">
                Updating previews…
              </Typography>
            </Stack>
          )}
        </Box>
        <Stack spacing={0.5} sx={{ minWidth: 220 }}>
          <Typography variant="caption" color="text.secondary">
            Samples per group: {previewSamplesPerGroup}
          </Typography>
          <Slider
            size="small"
            min={1}
            max={4}
            step={1}
            value={previewSamplesPerGroup}
            onChange={(_, value) => {
              const numeric = Array.isArray(value) ? value[0] : value;
              setPreviewSamplesPerGroup(typeof numeric === "number" ? numeric : 1);
            }}
            valueLabelDisplay="auto"
          />
          {previewQuery.data?.group_sample_counts && (
            <Typography variant="caption" color="text.secondary">
              Rendered subjects:{" "}
              {study.groups
                .map((group) => `${group}=${previewQuery.data?.group_sample_counts[group] ?? 0}`)
                .join(" • ")}
            </Typography>
          )}
          {study.groups.length > 0 && (
            <Button variant="text" size="small" onClick={() => setShowOverrides((current) => !current)}>
              {showOverrides ? "Hide per-group overrides" : "Adjust per-group overrides"}
            </Button>
          )}
          {showOverrides && (
            <Stack spacing={0.75} sx={{ border: "1px solid rgba(15,23,42,0.08)", borderRadius: 1, p: 1 }}>
              {study.groups.map((group) => (
                <TextField
                  key={`override-${group}`}
                  size="small"
                  type="number"
                  label={group}
                  inputProps={{ min: 1, max: 6 }}
                  value={previewGroupOverrides[group] ?? ""}
                  onChange={(event) => {
                    if (!event.target.value) {
                      setPreviewGroupOverride(group, null);
                      return;
                    }
                    const numeric = Number(event.target.value);
                    if (Number.isNaN(numeric)) {
                      return;
                    }
                    setPreviewGroupOverride(group, numeric);
                  }}
                />
              ))}
              <Button variant="text" size="small" onClick={resetPreviewGroupOverrides}>
                Reset overrides
              </Button>
            </Stack>
          )}
        </Stack>
        <Stack direction="row" spacing={1} alignItems="center">
          <Button
            variant="contained"
            size="small"
            startIcon={<RefreshIcon fontSize="small" />}
            onClick={handleRefreshPreviews}
            disabled={!study || refreshMutation.isPending}
          >
            {refreshMutation.isPending ? "Refreshing…" : "Refresh previews"}
          </Button>
          <Button
            variant="text"
            size="small"
            onClick={handleFlushCache}
            disabled={!study || clearCacheMutation.isPending}
          >
            {clearCacheMutation.isPending ? "Flushing…" : "Flush cache"}
          </Button>
        </Stack>
      </Box>
      {nd2Unavailable && (
        <Alert severity="warning">
          The original ND2 directory{nd2Source ? ` (${nd2Source})` : ""} is offline. Remount it and click “Refresh previews” to rebuild overlays.
        </Alert>
      )}
      {refreshMutation.isSuccess && (
        <Typography variant="caption" color="text.secondary">
          Regenerated {refreshMutation.data?.removed_directories.length ?? 0} threshold folders for the current sliders.
        </Typography>
      )}
      {refreshMutation.isError && (
        <Typography variant="caption" color="error">
          Unable to refresh previews. Confirm the ND2 drive is mounted.
        </Typography>
      )}
      {clearCacheMutation.isSuccess && (
        <Typography variant="caption" color="text.secondary">
          Flushed {clearCacheMutation.data?.removed_directories.length ?? 0} cache folders.
        </Typography>
      )}
      {clearCacheMutation.isError && (
        <Typography variant="caption" color="error">
          Failed to flush cached previews.
        </Typography>
      )}

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: `max-content repeat(${groupedPreviews.length}, minmax(${imageWidth}px, 1fr))`,
          columnGap: 1,
          alignItems: "end"
        }}
      >
        <Box />
        {groupedPreviews.map((column) => (
          <Tooltip
            key={`header-${column.key}`}
            title={
              <Stack spacing={0.25}>
                <Typography variant="caption">Subject: {column.subjectId}</Typography>
                <Typography variant="caption">File: {column.filename}</Typography>
              </Stack>
            }
            placement="top"
          >
            <Typography variant="subtitle2" noWrap>
              {column.group}
            </Typography>
          </Tooltip>
        ))}
      </Box>
      <Stack spacing={1.5}>
        {metricsWithData.map((metric) => (
          <Stack key={`metric-${metric.id}`} spacing={0.75}>
            <Typography
              variant="subtitle2"
              color={metric.id === selectedMetric ? "primary" : "text.primary"}
            >
              {metric.title}
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: `max-content repeat(${groupedPreviews.length}, minmax(${imageWidth}px, 1fr))`,
                columnGap: 1,
                rowGap: 0.75,
                alignItems: "start"
              }}
            >
              {variantOrder.map((variant) => (
                <Fragment key={`${metric.id}-${variant}`}>
                  <Box
                    sx={{
                      minHeight: imageHeight,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                    >
                      {variantLabels[variant]}
                    </Typography>
                  </Box>
                  {groupedPreviews.map((column) => {
                    const metricBucket = column.metrics.get(metric.id);
                    const images = metricBucket?.get(variant) ?? [];
                    const isReference = referenceGroup ? column.group === referenceGroup : false;
                    const highlighted = !activeGroups || activeGroups.has(column.group);

                    return (
                      <Box
                        key={`${column.key}-${metric.id}-${variant}`}
                        sx={{
                          minHeight: imageHeight,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: images.length > 0 ? "flex-start" : "center",
                          gap: 0.5,
                          opacity: highlighted ? 1 : 0.4,
                          borderBottom: isReference && variant === "overlay" ? "2px solid rgba(37,99,235,0.25)" : "none",
                          pb: 0.25
                        }}
                      >
                        {images.length > 0 ? (
                          images.map((image) => {
                            const cacheBust = image.variant === "raw" ? "" : `&v=${encodeURIComponent(token)}`;
                            const src = `${apiBase}/studies/${study.study_id}/preview-file?path=${encodeURIComponent(
                              image.image_path
                            )}${cacheBust}`;
                            const downloadName = `${image.group}_${image.metric}_${image.variant}.png`;
                            return (
                              <Box
                                key={`${column.key}-${metric.id}-${previewKey(image)}`}
                                sx={{
                                  position: "relative",
                                  width: imageWidth,
                                  height: imageHeight,
                                  borderRadius: 0.5,
                                  backgroundColor: "#fff"
                                }}
                              >
                                <Box
                                  component="img"
                                  src={src}
                                  alt={`${column.group} ${metric.title} ${variantLabels[variant]}`}
                                  sx={{
                                    width: "100%",
                                    height: "100%",
                                    objectFit: "contain",
                                    borderRadius: 0.5
                                  }}
                                />
                                <Tooltip title="Download PNG">
                                  <IconButton
                                    component="a"
                                    href={src}
                                    download={downloadName}
                                    size="small"
                                    sx={{
                                      position: "absolute",
                                      top: 4,
                                      right: 4,
                                      backgroundColor: "rgba(15,23,42,0.6)",
                                      color: "#fff",
                                      "&:hover": {
                                        backgroundColor: "rgba(15,23,42,0.8)"
                                      }
                                    }}
                                  >
                                    <DownloadIcon fontSize="inherit" />
                                  </IconButton>
                                </Tooltip>
                              </Box>
                            );
                          })
                        ) : (
                          <Typography variant="caption" color="text.secondary">
                            —
                          </Typography>
                        )}
                      </Box>
                    );
                  })}
                </Fragment>
              ))}
            </Box>
          </Stack>
        ))}
      </Stack>
    </Stack>
  );
}
