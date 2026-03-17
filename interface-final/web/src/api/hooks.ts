import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { api, apiClient } from "./client";
import type {
  AnalyzeResponse,
  ConfigAutoGroupResponse,
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
  ChannelDefinition,
  ChannelUpdateResponse,
  PixelSizeUpdateResponse,
  PreviewDownloadResponse
} from "./types";

const DEFAULT_ANALYSIS_MODE = "positive_area_percent" as const;

export type ChannelRangePayload = Partial<Record<"channel_1" | "channel_2" | "channel_3", { vmin: number; vmax: number }>>;
type PreviewPrioritySubjectPayload = { group: string; subject_id: string; filename: string };

export function useConfigScan() {
  return useMutation({
    mutationFn: (payload: { input_dir: string; recursive?: boolean; subject_strategy?: "per_file" | "auto" }) =>
      api.post<ConfigScanResponse>("/config/scan", payload)
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
      channel_definitions?: ChannelDefinition[];
    }) => api.post<ConfigCreateResponse>("/config/create", payload)
  });
}

export function useConfigAutoGroups() {
  return useMutation({
    mutationFn: (payload: { input_dir: string; instructions: string; model?: string }) =>
      api.post<ConfigAutoGroupResponse>("/config/auto-groups", payload)
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
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === "queued" || state === "running" || state === undefined ? 1000 : false;
    }
  });
}

export function useAnalysisQuery(
  studyId: string | null,
  thresholds: Record<string, number>
) {
  return useQuery({
    queryKey: [
      "analysis",
      studyId,
      thresholds.channel_1,
      thresholds.channel_2,
      thresholds.channel_3,
      DEFAULT_ANALYSIS_MODE
    ],
    queryFn: () =>
      api.post<AnalyzeResponse>(`/studies/${studyId}/analyze`, {
        thresholds,
        analysis_mode: DEFAULT_ANALYSIS_MODE
      }),
    enabled: Boolean(studyId),
    placeholderData: keepPreviousData
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
      options.significanceDisplay,
      DEFAULT_ANALYSIS_MODE
    ],
    queryFn: () =>
      api.post<StatisticsResponse>(`/studies/${studyId}/statistics`, {
        thresholds,
        comparison_mode: options.comparisonMode,
        reference_group: options.referenceGroup,
        comparison_pairs: options.comparisonPairs,
        test_type: options.testType,
        significance_display: options.significanceDisplay,
        analysis_mode: DEFAULT_ANALYSIS_MODE
      }),
    enabled: Boolean(studyId) && options.enabled,
    placeholderData: keepPreviousData
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
  rangeKey?: string,
  options?: {
    enabled?: boolean;
    phase?: "focused" | "full" | "default";
    revisionKey?: string;
    prioritySubjects?: PreviewPrioritySubjectPayload[];
    preferGeneratedAssets?: boolean;
  }
) {
  const metricsKey = metrics.length ? metrics.join("|") : "default";
  const phase = options?.phase ?? "default";
  const revisionKey = options?.revisionKey ?? "default";
  const preferGeneratedAssets = options?.preferGeneratedAssets ?? true;
  const limitsKey =
    groupLimits && Object.keys(groupLimits).length
      ? Object.entries(groupLimits)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([group, value]) => `${group}:${value}`)
          .join("|")
      : "none";
  const prioritySubjects = options?.prioritySubjects ?? [];
  const priorityKey = prioritySubjects.length
    ? prioritySubjects
        .map((subject) => `${subject.group}|${subject.subject_id}|${subject.filename}`)
        .sort()
        .join("||")
    : "none";
  return useQuery({
    queryKey: [
      "previews",
      studyId,
      phase,
      thresholds.channel_1,
      thresholds.channel_2,
      thresholds.channel_3,
      metricsKey,
      groups?.join("|") ?? "all",
      sampleCount,
      limitsKey,
      rangeKey ?? "default",
      revisionKey,
      priorityKey,
      preferGeneratedAssets ? "generated" : "source"
    ],
    queryFn: ({ signal }) =>
      api.post<PreviewResponse>(`/studies/${studyId}/previews`, {
        thresholds,
        metrics,
        groups,
        max_samples_per_group: sampleCount,
        group_sample_limits: groupLimits,
        channel_ranges: channelRanges,
        revision_key: revisionKey,
        priority_subjects: prioritySubjects,
        prefer_generated_assets: preferGeneratedAssets
      }, {
        signal
      }),
    enabled: options?.enabled ?? Boolean(studyId),
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

export function useUpdateChannels(studyId: string | null) {
  return useMutation({
    mutationFn: (channels: ChannelDefinition[]) => {
      if (!studyId) {
        return Promise.reject(new Error("Study not loaded"));
      }
      return api.post<ChannelUpdateResponse>(`/studies/${studyId}/channels`, {
        channels
      });
    }
  });
}
