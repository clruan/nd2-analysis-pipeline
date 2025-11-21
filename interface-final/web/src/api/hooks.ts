import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { api, apiClient } from "./client";
import type {
  AnalyzeResponse,
  ConfigCreateResponse,
  ConfigReadResponse,
  ConfigScanResponse,
  DownloadResponse,
  LoadedStudy,
  PreviewResponse,
  PreviewClearResponse,
  RunStatus,
  StatisticsResponse,
  UploadResponse,
  RatioUpdateResponse,
  RatioDefinition,
  PixelSizeUpdateResponse,
  PreviewDownloadResponse
} from "./types";

export type ChannelRangePayload = Partial<Record<"channel_1" | "channel_2" | "channel_3", { vmin: number; vmax: number }>>;

export function useConfigScan() {
  return useMutation({
    mutationFn: (payload: { input_dir: string; recursive?: boolean }) => api.post<ConfigScanResponse>("/config/scan", payload)
  });
}

export function useConfigCreate() {
  return useMutation({
    mutationFn: (payload: {
      input_dir: string;
      study_name: string;
      groups: Record<string, string[]>;
      pixel_size_um?: number;
      thresholds?: Record<string, Record<string, number>>;
      output_path?: string;
      ratios?: RatioDefinition[];
    }) => api.post<ConfigCreateResponse>("/config/create", payload)
  });
}

export function useConfigRead() {
  return useMutation({
    mutationFn: (payload: { path: string }) => api.get<ConfigReadResponse>("/config/read", payload)
  });
}

export function useThresholdRun() {
  return useMutation({
    mutationFn: (payload: {
      input_dir: string;
      config_path: string;
      output_path?: string;
      is_3d?: boolean;
      marker?: string;
      n_jobs?: number;
      max_threshold?: number;
    }) => api.post<RunStatus>("/runs/threshold", payload)
  });
}

export function useLoadStudy() {
  return useMutation({
    mutationFn: (payload: { file_path: string; input_dir_override?: string }) => api.post<LoadedStudy>("/studies/load", payload)
  });
}

export function useRunStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["run-status", jobId],
    queryFn: () => api.get<RunStatus>(`/runs/${jobId}`),
    enabled: Boolean(jobId)
  });
}

export function useAnalysisQuery(studyId: string | null, thresholds: Record<string, number>) {
  return useQuery({
    queryKey: ["analysis", studyId, thresholds.channel_1, thresholds.channel_2, thresholds.channel_3],
    queryFn: () =>
      api.post<AnalyzeResponse>(`/studies/${studyId}/analyze`, {
        thresholds
      }),
    enabled: Boolean(studyId)
  });
}

export function useStatisticsQuery(
  studyId: string | null,
  thresholds: Record<string, number>,
  options: {
    enabled: boolean;
    comparisonMode: "all_vs_one" | "pairs" | "all_pairs";
    referenceGroup?: string | null;
    comparisonPairs?: string[][];
    testType: "anova_parametric" | "anova_non_parametric" | "t_test";
    significanceDisplay: "stars" | "p_values";
  }
) {
  const pairsKey =
    options.comparisonMode === "pairs"
      ? (options.comparisonPairs ?? [])
          .map((pair) => pair.slice().sort().join("::"))
          .sort()
          .join("|")
      : "none";
  return useQuery({
    queryKey: [
      "statistics",
      studyId,
      thresholds.channel_1,
      thresholds.channel_2,
      thresholds.channel_3,
      options.comparisonMode,
      options.referenceGroup ?? "none",
      pairsKey,
      options.testType,
      options.significanceDisplay
    ],
    queryFn: () =>
      api.post<StatisticsResponse>(`/studies/${studyId}/statistics`, {
        thresholds,
        comparison_mode: options.comparisonMode,
        reference_group: options.referenceGroup,
        comparison_pairs: options.comparisonPairs,
        test_type: options.testType,
        significance_display: options.significanceDisplay
      }),
    enabled: Boolean(studyId) && options.enabled
  });
}

export function usePreviewQuery(
  studyId: string | null,
  thresholds: Record<string, number>,
  metrics: string[],
  sampleCount: number,
  groups?: string[],
  groupLimits?: Record<string, number>,
  channelRanges?: ChannelRangePayload,
  rangeKey?: string
) {
  const metricsKey = metrics.length ? metrics.join("|") : "default";
  const limitsKey =
    groupLimits && Object.keys(groupLimits).length
      ? Object.entries(groupLimits)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([group, value]) => `${group}:${value}`)
          .join("|")
      : "none";
  return useQuery({
    queryKey: [
      "previews",
      studyId,
      thresholds.channel_1,
      thresholds.channel_2,
      thresholds.channel_3,
      metricsKey,
      groups?.join("|") ?? "all",
      sampleCount,
      limitsKey,
      rangeKey ?? "default"
    ],
    queryFn: () =>
      api.post<PreviewResponse>(`/studies/${studyId}/previews`, {
        thresholds,
        metrics,
        groups,
        max_samples_per_group: sampleCount,
        group_sample_limits: groupLimits,
        channel_ranges: channelRanges
      }),
    enabled: Boolean(studyId),
    placeholderData: keepPreviousData
  });
}

export function useDownloadMutation(studyId: string | null) {
  return useMutation({
    mutationFn: (thresholds: Record<string, number>) =>
      api.post<DownloadResponse>(`/studies/${studyId}/downloads/current`, {
        thresholds
      })
  });
}

export function useClearPreviewsMutation(studyId: string | null) {
  return useMutation({
    mutationFn: (payload?: { scope?: "thresholds" | "all"; threshold_key?: string }) => {
      if (!studyId) {
        return Promise.reject(new Error("Study not loaded"));
      }
      return api.post<PreviewClearResponse>(`/studies/${studyId}/previews/clear`, payload ?? { scope: "thresholds" });
    }
  });
}

export function useFileUpload() {
  return useMutation({
    mutationFn: async ({ category, file }: { category: "config" | "threshold_results"; file: File }) => {
      const data = new FormData();
      data.append("category", category);
      data.append("file", file);
      const response = await apiClient.post<UploadResponse>("/uploads", data, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      return response.data;
    }
  });
}

export function usePreviewDownload(studyId: string | null) {
  return useMutation({
    mutationFn: (payload: {
      group: string;
      subject_id: string;
      filename: string;
      thresholds: Record<string, number>;
      panel_order: string[];
      channel_ranges?: ChannelRangePayload;
      scale_bar_um?: number;
    }) => {
      if (!studyId) {
        return Promise.reject(new Error("Study not loaded"));
      }
      return api.post<PreviewDownloadResponse>(`/studies/${studyId}/previews/render`, payload);
    }
  });
}

export function usePixelSizeUpdate(studyId: string | null) {
  return useMutation({
    mutationFn: (pixelSize?: number | null) => {
      if (!studyId) {
        return Promise.reject(new Error("Study not loaded"));
      }
      return api.post<PixelSizeUpdateResponse>(`/studies/${studyId}/pixel-size`, {
        pixel_size_um: typeof pixelSize === "number" ? pixelSize : null
      });
    }
  });
}

export function useUpdateRatios(studyId: string | null) {
  return useMutation({
    mutationFn: (ratios: RatioDefinition[]) => {
      if (!studyId) {
        return Promise.reject(new Error("Study not loaded"));
      }
      return api.post<RatioUpdateResponse>(`/studies/${studyId}/ratios`, {
        ratios
      });
    }
  });
}
