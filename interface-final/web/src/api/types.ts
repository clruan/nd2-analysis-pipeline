export interface ConfigScanResponse {
  study_name: string;
  input_dir: string;
  nd2_files: string[];
  groups: Array<{
    group_name: string;
    subjects: Array<{
      subject_id: string;
      replicates: Array<{ filename: string; absolute_path: string }>;
    }>;
  }>;
}

export interface ConfigCreateResponse {
  config_path: string;
  study_name: string;
  groups: Record<string, string[]>;
}

export interface ConfigReadResponse {
  config_path: string;
  study_name: string;
  groups: Record<string, string[]>;
  pixel_size_um?: number | null;
  thresholds?: Record<string, Record<string, number>> | null;
  ratios?: RatioDefinition[] | null;
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
  Channel_1_area: number;
  Channel_2_area: number;
  Channel_3_area: number;
  ratios?: Record<string, number>;
}

export interface IndividualImageRecord {
  group: string;
  mouse_id: string;
  filename: string;
  channel_1_area: number;
  channel_2_area: number;
  channel_3_area: number;
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
