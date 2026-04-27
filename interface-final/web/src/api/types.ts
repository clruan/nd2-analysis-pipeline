export interface ConfigScanResponse {
  study_name: string;
  input_dir: string;
  nd2_files: string[];
  channel_definitions?: ChannelDefinition[] | null;
  groups: Array<{
    group_name: string;
    subjects: Array<{
      subject_id: string;
      replicates: Array<{ filename: string; absolute_path: string }>;
    }>;
  }>;
}

export interface ChannelDefinition {
  channel: number;
  label: string;
  color: string;
}

export interface ConfigCreateResponse {
  config_path: string;
  study_name: string;
  groups: Record<string, string[]>;
  channel_definitions?: ChannelDefinition[] | null;
}

export interface ConfigAutoGroupResponse {
  groups: Record<string, string[]>;
  model: string;
  notes?: string | null;
}

export interface ConfigReadResponse {
  config_path: string;
  study_name: string;
  groups: Record<string, string[]>;
  pixel_size_um?: number | null;
  thresholds?: Record<string, Record<string, number>> | null;
  ratios?: RatioDefinition[] | null;
  channel_definitions?: ChannelDefinition[] | null;
}

export interface RunStatus {
  job_id: string;
  state: "queued" | "running" | "succeeded" | "failed";
  message?: string;
  input_dir?: string | null;
  config_path?: string | null;
  output_path?: string | null;
  study_name?: string | null;
  started_at: string;
  completed_at?: string | null;
  latest_source_mtime?: string | null;
  source_hash?: string | null;
  progress_completed: number;
  progress_total?: number | null;
}

export interface LoadedStudy {
  study_id: string;
  study_name: string;
  source_path: string;
  groups: string[];
  mice_count: number;
  image_count: number;
  nd2_root: string;
  nd2_available: boolean;
  ratio_definitions: RatioDefinition[];
  channel_definitions: ChannelDefinition[];
  channel_limits: Record<string, number>;
  max_threshold: number;
  pixel_size_um?: number | null;
}

export interface RatioDefinition {
  id: string;
  label: string;
  numerator_channel: number;
  denominator_channel: number;
}

export interface MouseAverageRecord {
  Group: string;
  MouseID: string;
  channel_areas: Record<string, number>;
  ratios?: Record<string, number>;
}

export interface IndividualImageRecord {
  group: string;
  mouse_id: string;
  filename: string;
  channel_areas: Record<string, number>;
  ratios: Record<string, number>;
  replicate_index: number;
}

export interface AnalyzeResponse {
  study_id: string;
  thresholds: Record<string, number>;
  mouse_averages: MouseAverageRecord[];
  individual_images: IndividualImageRecord[];
}

export interface StatisticsResponse {
  statistics: Record<string, unknown>;
  thresholds: Record<string, number>;
  test_type_used: string;
  significance_display: string;
  ratios: RatioDefinition[];
}

export interface PreviewImage {
  variant: "raw" | "mask" | "overlay";
  metric: string;
  channel?: number | null;
  group: string;
  subject_id: string;
  filename: string;
  image_path: string;
  cache_token: string;
}

export interface PreviewResponse {
  study_id: string;
  generated_at: string;
  images: PreviewImage[];
  nd2_available: boolean;
  nd2_source?: string | null;
  max_samples_per_group: number;
  ratio_definitions: RatioDefinition[];
  group_sample_counts: Record<string, number>;
}

export interface PreviewDownloadResponse {
  image_path: string;
  panel_order: string[];
  composite_channels: number[];
}

export interface PreviewClearResponse {
  removed_directories: string[];
}

export interface DownloadResponse {
  download_path: string;
  generated_at: string;
}

export interface PixelSizeUpdateResponse {
  pixel_size_um?: number | null;
}

export interface UploadResponse {
  stored_path: string;
  original_name: string;
  category: string;
}

export interface RatioUpdateResponse {
  ratios: RatioDefinition[];
}

export interface ChannelUpdateResponse {
  channels: ChannelDefinition[];
}
