import { useEffect, useMemo, useRef, useState } from "react";
import { isAxiosError } from "axios";
import {
  Alert,
  Box,
  Button,
  Chip,
  Checkbox,
  Collapse,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Slider,
  Stack,
  Switch,
  TextField,
  Typography
} from "@mui/material";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useAppStore } from "../state/useAppStore";
import {
  useConfigAutoGroups,
  useConfigCreate,
  useConfigRead,
  useConfigScan,
  useFileUpload,
  useLoadStudy,
  useRunStatus,
  useThresholdRun,
  useUpdateChannels,
  useUpdateRatios,
  usePixelSizeUpdate
} from "../api/hooks";
import type { ChannelDefinition, ConfigScanResponse, RatioDefinition } from "../api/types";
import {
  buildDefaultChannelDefinitions,
  DEFAULT_CHANNEL_DEFINITIONS,
  DEFAULT_RATIO_DEFINITIONS,
  normalizeChannelDefinitions
} from "../constants/metrics";

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

type PalettePreset = {
  id: string;
  label: string;
  description: string;
  colors: string[];
};

const palettePresets: PalettePreset[] = [
  {
    id: "vivid",
    label: "Vivid contrast",
    description: "High-contrast default palette.",
    colors: defaultPalette
  },
  {
    id: "colorBlind",
    label: "Color-blind friendly",
    description: "Wong palette tuned for most common color deficiencies.",
    colors: ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#999999"]
  },
  {
    id: "monotoneBlue",
    label: "Monotone blues",
    description: "Single hue gradient for subtle differences.",
    colors: ["#0f172a", "#1e3a8a", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
  },
  {
    id: "warmSequential",
    label: "Warm sequential",
    description: "Orange sequential palette for ordered groups.",
    colors: ["#7c2d12", "#9a3412", "#c2410c", "#ea580c", "#f97316", "#fb923c", "#fdba74", "#fed7aa"]
  }
];

type StepId = "project" | "configuration" | "analysis";
type GuideState = "completed" | "active" | "upcoming";
type ModuleId =
  | "scan"
  | "config_builder"
  | "analysis_controls"
  | "visualization_settings"
  | "threshold_generation"
  | "study_loader";
type ChannelKey = string;
type ChannelWindowDraft = { min: string; threshold: string; max: string };
type ChannelWindowCommitSource = "min" | "threshold" | "max";

const baseSectionSx = {
  borderRadius: 2,
  p: 2,
  transition: "border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease"
} as const;

const guideStateStyles: Record<GuideState, { borderColor: string; backgroundColor: string; boxShadow: string }> = {
  completed: {
    borderColor: "rgba(34,197,94,0.6)",
    backgroundColor: "rgba(34,197,94,0.06)",
    boxShadow: "none"
  },
  active: {
    borderColor: "rgba(37,99,235,0.7)",
    backgroundColor: "rgba(37,99,235,0.05)",
    boxShadow: "0 0 0 3px rgba(37,99,235,0.15)"
  },
  upcoming: {
    borderColor: "rgba(15,23,42,0.12)",
    backgroundColor: "#ffffff",
    boxShadow: "none"
  }
};

const sectionStylesForState = (state: GuideState) => ({
  ...baseSectionSx,
  border: `1px solid ${guideStateStyles[state].borderColor}`,
  backgroundColor: guideStateStyles[state].backgroundColor,
  boxShadow: guideStateStyles[state].boxShadow
});

const normalizeGroupMapping = (groups: Record<string, string[]>): Record<string, string[]> => {
  const normalized: Record<string, string[]> = {};
  Object.entries(groups).forEach(([group, subjects]) => {
    const cleaned = subjects.map((subject) => subject.trim()).filter(Boolean);
    normalized[group] = Array.from(new Set(cleaned)).sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
    );
  });
  return normalized;
};

const pairToken = (first: string, second: string) =>
  [first, second]
    .sort((a, b) => a.localeCompare(b))
    .join("::");

const getErrorMessage = (error: unknown) => {
  if (!error) return "Unknown error";
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const message = (detail as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
    if (error.response?.statusText) return error.response.statusText;
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : JSON.stringify(error);
};

const clampIntensity = (value: number, max: number) => Math.max(0, Math.min(max, Math.round(value)));

const parseIntensityDraft = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return clampIntensity(parsed, Number.MAX_SAFE_INTEGER);
};

const buildChannelWindowDrafts = (
  ranges: Record<string, [number, number]>,
  thresholds: Record<string, number>
): Record<string, ChannelWindowDraft> =>
  Object.keys(ranges).reduce<Record<string, ChannelWindowDraft>>((acc, channel) => {
    acc[channel] = {
      min: String(ranges[channel][0]),
      threshold: String(thresholds[channel] ?? 0),
      max: String(ranges[channel][1])
    };
    return acc;
  }, {});

const sliderValueFromDraft = (
  range: [number, number],
  threshold: number,
  draft: ChannelWindowDraft
): [number, number, number] => {
  const min = parseIntensityDraft(draft.min);
  const thresholdValue = parseIntensityDraft(draft.threshold);
  const max = parseIntensityDraft(draft.max);
  if (min === null || thresholdValue === null || max === null) {
    return [range[0], threshold, range[1]];
  }
  const sorted = [min, thresholdValue, max].sort((a, b) => a - b) as [number, number, number];
  return sorted;
};

const getActiveSliderThumbIndex = () => {
  if (typeof document === "undefined") {
    return 1;
  }
  const raw = document.activeElement?.getAttribute("data-index");
  const parsed = raw === null ? Number.NaN : Number(raw);
  return Number.isFinite(parsed) ? parsed : 1;
};

export default function LeftPanel() {
  const {
    study,
    thresholds,
    statisticsEnabled,
    statisticsSettings,
    plotSettings,
    ratioDefinitions,
    channelDefinitions,
    setThreshold,
    setThresholdControlHovered,
    setStudy,
    setStatisticsEnabled,
    setComparisonMode,
    setReferenceGroup,
    addComparisonPair,
    removeComparisonPair,
    clearComparisonPairs,
    setSignificanceDisplay,
    setTestType,
    setPlotTitle,
    setPlotFontSize,
    setPaletteColor,
    setPalette,
    setJitterEnabled,
    setJitterWidth,
    setRatioDefinitions,
    setChannelDefinitions,
    previewSamplesPerGroup,
    setPreviewSamplesPerGroup,
    previewChannelRanges,
    setPreviewChannelRange,
    resetPreviewChannelRanges,
    updateStudy
  } = useAppStore();
  const [inputDir, setInputDir] = useState("");
  const [scanSubjectStrategy, setScanSubjectStrategy] = useState<"per_file" | "auto">("per_file");
  const [configPath, setConfigPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [resultsPath, setResultsPath] = useState("");
  const [activeStage, setActiveStage] = useState<StepId>("project");
  const [configOriginalName, setConfigOriginalName] = useState<string | null>(null);
  const [resultsOriginalName, setResultsOriginalName] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<ConfigScanResponse | null>(null);
  const [groupsJson, setGroupsJson] = useState("{}");
  const [pixelSize, setPixelSize] = useState("");
  const [configError, setConfigError] = useState<string | null>(null);
  const [autoGroupInstructions, setAutoGroupInstructions] = useState("");
  const [autoGroupModel, setAutoGroupModel] = useState("");
  const [groupMap, setGroupMap] = useState<Record<string, string[]>>({});
  const [newGroupName, setNewGroupName] = useState("");
  const [subjectInputs, setSubjectInputs] = useState<Record<string, string>>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [ratioDrafts, setRatioDrafts] = useState<RatioDefinition[]>(DEFAULT_RATIO_DEFINITIONS);
  const [channelDrafts, setChannelDrafts] = useState<ChannelDefinition[]>(DEFAULT_CHANNEL_DEFINITIONS);
  const [studyRatioDrafts, setStudyRatioDrafts] = useState<RatioDefinition[]>(DEFAULT_RATIO_DEFINITIONS);
  const [studyChannelDrafts, setStudyChannelDrafts] = useState<ChannelDefinition[]>(DEFAULT_CHANNEL_DEFINITIONS);
  const [palettePresetOverride, setPalettePresetOverride] = useState<string | null>(null);
  const [studyPixelSize, setStudyPixelSize] = useState("");
  const [pixelSizeError, setPixelSizeError] = useState<string | null>(null);
  const [collapsedModules, setCollapsedModules] = useState<Record<ModuleId, boolean>>({
    scan: false,
    config_builder: false,
    analysis_controls: false,
    visualization_settings: false,
    threshold_generation: false,
    study_loader: false,
  });
  const [channelWindowDrafts, setChannelWindowDrafts] = useState<Record<string, ChannelWindowDraft>>(() =>
    buildChannelWindowDrafts(previewChannelRanges, thresholds)
  );

  const scanMutation = useConfigScan();
  const autoGroupsMutation = useConfigAutoGroups();
  const configMutation = useConfigCreate();
  const configReadMutation = useConfigRead();
  const runMutation = useThresholdRun();
  const loadMutation = useLoadStudy();
  const statusQuery = useRunStatus(jobId);
  const uploadMutation = useFileUpload();
  const updateRatiosMutation = useUpdateRatios(study?.study_id ?? null);
  const updateChannelsMutation = useUpdateChannels(study?.study_id ?? null);
  const pixelSizeMutation = usePixelSizeUpdate(study?.study_id ?? null);
  const configInputRef = useRef<HTMLInputElement>(null);
  const resultsInputRef = useRef<HTMLInputElement>(null);
  const activeSliderThumbRef = useRef<Record<string, number>>({});

  useEffect(() => {
    return () => {
      setThresholdControlHovered(false);
    };
  }, [setThresholdControlHovered]);

  const studyGroups = useMemo(() => study?.groups ?? [], [study?.groups]);
  const pairOptions = useMemo<Array<[string, string]>>(() => {
    const options: Array<[string, string]> = [];
    for (let index = 0; index < studyGroups.length; index += 1) {
      for (let other = index + 1; other < studyGroups.length; other += 1) {
        options.push([studyGroups[index], studyGroups[other]]);
      }
    }
    return options;
  }, [studyGroups]);
  const selectedPairTokens = useMemo(
    () => new Set(statisticsSettings.comparisonPairs.map(([first, second]) => pairToken(first, second))),
    [statisticsSettings.comparisonPairs]
  );

  const handleStudyPixelSizeSave = () => {
    if (!study) return;
    setPixelSizeError(null);
    const trimmed = studyPixelSize.trim();
    let value: number | null = null;
    if (trimmed !== "") {
      const parsed = Number(trimmed);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        setPixelSizeError("Enter a positive value or leave blank to clear.");
        return;
      }
      value = parsed;
    }
    pixelSizeMutation.mutate(value, {
      onSuccess: (response) => {
        updateStudy({ pixel_size_um: response.pixel_size_um ?? null });
        if (!response.pixel_size_um) {
          setStudyPixelSize("");
        } else {
          setStudyPixelSize(String(response.pixel_size_um));
        }
      },
      onError: (error) => {
        setPixelSizeError(getErrorMessage(error));
      }
    });
  };

  useEffect(() => {
    setGroupsJson(JSON.stringify(groupMap, null, 2));
  }, [groupMap]);

  useEffect(() => {
    setStudyRatioDrafts(ratioDefinitions);
  }, [ratioDefinitions, study?.study_id]);

  useEffect(() => {
    const normalized = normalizeChannelDefinitions(channelDefinitions);
    setStudyChannelDrafts(normalized);
  }, [channelDefinitions, study?.study_id]);

  useEffect(() => {
    setPalettePresetOverride(null);
  }, [study?.study_id]);

  useEffect(() => {
    if (typeof study?.pixel_size_um === "number" && Number.isFinite(study.pixel_size_um)) {
      setStudyPixelSize(String(study.pixel_size_um));
    } else {
      setStudyPixelSize("");
    }
    setPixelSizeError(null);
  }, [study?.pixel_size_um]);

  useEffect(() => {
    setSubjectInputs((prev) => {
      const next: Record<string, string> = {};
      Object.keys(groupMap).forEach((group) => {
        if (prev[group]) {
          next[group] = prev[group];
        }
      });
      return next;
    });
  }, [groupMap]);

  const scanDefaults = useMemo(() => {
    if (!scanResult) return {};
    const defaults: Record<string, string[]> = {};
    scanResult.groups.forEach((group) => {
      defaults[group.group_name] = group.subjects.map((subject) => subject.subject_id);
    });
    return defaults;
  }, [scanResult]);

  const scannedSubjects = useMemo(
    () =>
      (scanResult?.groups ?? [])
        .flatMap((group) =>
          group.subjects.map((subject) => ({
            subjectId: subject.subject_id,
            detectedGroup: group.group_name,
            replicates: subject.replicates.map((replicate) => replicate.filename),
            replicateCount: subject.replicates.length
          }))
        )
        .sort((a, b) => a.subjectId.localeCompare(b.subjectId, undefined, { numeric: true, sensitivity: "base" })),
    [scanResult]
  );

  const assignedGroupBySubject = useMemo(() => {
    const next: Record<string, string> = {};
    scannedSubjects.forEach((subject) => {
      const mappedGroup = Object.entries(groupMap).find(([, subjects]) => subjects.includes(subject.subjectId))?.[0];
      next[subject.subjectId] = mappedGroup ?? subject.detectedGroup;
    });
    return next;
  }, [groupMap, scannedSubjects]);

  const assignSubjectToGroup = (subjectId: string, targetGroup: string) => {
    if (!targetGroup.trim()) return;
    setGroupMap((prev) => {
      const next: Record<string, string[]> = Object.fromEntries(
        Object.entries(prev).map(([group, subjects]) => [group, subjects.filter((item) => item !== subjectId)])
      );
      if (!next[targetGroup]) {
        next[targetGroup] = [];
      }
      next[targetGroup] = Array.from(new Set([...next[targetGroup], subjectId])).sort((a, b) =>
        a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
      );
      return next;
    });
    setConfigError(null);
  };

  const statsControlsDisabled = !statisticsEnabled || studyGroups.length === 0;
  const getChannelLimit = (channel: string) =>
    Math.max(1, study?.channel_limits?.[channel] ?? study?.max_threshold ?? previewChannelRanges[channel]?.[1] ?? 4095);

  useEffect(() => {
    if (!studyGroups.length) {
      return;
    }
    const nextPalette = { ...plotSettings.palette };
    let changed = false;
    studyGroups.forEach((group, index) => {
      if (!nextPalette[group]) {
        nextPalette[group] = defaultPalette[index % defaultPalette.length];
        changed = true;
      }
    });
    if (changed) {
      setPalette(nextPalette);
    }
  }, [plotSettings.palette, setPalette, studyGroups]);

  const effectivePalette = useMemo(() => {
    const colors: Record<string, string> = { ...plotSettings.palette };
    studyGroups.forEach((group, index) => {
      if (!colors[group]) {
        colors[group] = defaultPalette[index % defaultPalette.length];
      }
    });
    return colors;
  }, [plotSettings.palette, studyGroups]);

  const detectedPalettePreset = useMemo(() => {
    if (!studyGroups.length) {
      return "custom";
    }
    const match = palettePresets.find((preset) =>
      studyGroups.every((group, index) => {
        const expected = preset.colors[index % preset.colors.length].toLowerCase();
        const actual = (effectivePalette[group] ?? "").toLowerCase();
        return expected === actual;
      })
    );
    return match?.id ?? "custom";
  }, [effectivePalette, studyGroups]);

  const palettePreset = palettePresetOverride ?? detectedPalettePreset;
  const activePalettePreset = palettePresets.find((preset) => preset.id === palettePreset) ?? null;

  const jitterEnabled = plotSettings.jitterEnabled;

  const togglePairSelection = (pair: [string, string]) => {
    const token = pairToken(pair[0], pair[1]);
    if (selectedPairTokens.has(token)) {
      removeComparisonPair(pair);
      return;
    }
    addComparisonPair(pair);
  };

  const handlePalettePresetChange = (value: string) => {
    if (value === "custom") {
      setPalettePresetOverride("custom");
      return;
    }
    const preset = palettePresets.find((entry) => entry.id === value);
    if (!preset || studyGroups.length === 0) {
      setPalettePresetOverride("custom");
      return;
    }
    const nextPalette = studyGroups.reduce<Record<string, string>>((acc, group, index) => {
      acc[group] = preset.colors[index % preset.colors.length];
      return acc;
    }, {});
    setPalette(nextPalette);
    setPalettePresetOverride(value);
  };

  const handleGroupColorChange = (group: string, color: string) => {
    setPaletteColor(group, color);
    setPalettePresetOverride("custom");
  };

  const handleAddGroup = () => {
    const trimmed = newGroupName.trim();
    if (!trimmed) {
      return;
    }
    setGroupMap((prev) => {
      if (prev[trimmed]) {
        return prev;
      }
      return { ...prev, [trimmed]: [] };
    });
    setNewGroupName("");
    setConfigError(null);
  };

  const handleAddSubject = (group: string, subjectId: string) => {
    const trimmed = subjectId.trim();
    if (!trimmed) return;
    setGroupMap((prev) => {
      const next = { ...prev };
      const existing = new Set(next[group] ?? []);
      existing.add(trimmed);
      next[group] = Array.from(existing).sort();
      return next;
    });
    setSubjectInputs((prev) => ({ ...prev, [group]: "" }));
    setConfigError(null);
  };

  const handleRemoveSubject = (group: string, subjectId: string) => {
    setGroupMap((prev) => {
      const next = { ...prev };
      next[group] = (next[group] ?? []).filter((id) => id !== subjectId);
      return next;
    });
  };

  const handleRemoveGroup = (group: string) => {
    setGroupMap((prev) => {
      const next = { ...prev };
      delete next[group];
      return next;
    });
    setSubjectInputs((prev) => {
      const next = { ...prev };
      delete next[group];
      return next;
    });
  };

  const sanitizeChannel = (value: number | string | undefined) => {
    if (value === undefined || value === null) {
      return undefined;
    }
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
      return undefined;
    }
    return Math.max(1, Math.round(numeric));
  };

  const updateRatioDraft = (
    setter: React.Dispatch<React.SetStateAction<RatioDefinition[]>>,
    index: number,
    patch: Partial<RatioDefinition>
  ) => {
    setter((prev) =>
      prev.map((ratio, idx) => {
        if (idx !== index) {
          return ratio;
        }
        const next: RatioDefinition = {
          ...ratio,
          ...patch
        };
        const numerator = sanitizeChannel(patch.numerator_channel);
        if (numerator !== undefined) {
          next.numerator_channel = numerator;
        }
        const denominator = sanitizeChannel(patch.denominator_channel);
        if (denominator !== undefined) {
          next.denominator_channel = denominator;
        }
        if (next.numerator_channel === next.denominator_channel) {
          return ratio;
        }
        if (!next.id) {
          next.id = ratio.id || `ratio_${idx}`;
        }
        if (!next.label) {
          next.label = ratio.label;
        }
        return next;
      })
    );
  };

  const addRatioDraft = (setter: React.Dispatch<React.SetStateAction<RatioDefinition[]>>) => {
    setter((prev) => [
      ...prev,
      {
        id: `custom_ratio_${Date.now()}`,
        label: `Custom ${prev.length + 1}`,
        numerator_channel: 1,
        denominator_channel: Math.max(
          2,
          prev.reduce((maxChannel, ratio) => Math.max(maxChannel, ratio.numerator_channel, ratio.denominator_channel), 3)
        )
      }
    ]);
  };

  const handleConfigRatioChange = (index: number, patch: Partial<RatioDefinition>) =>
    updateRatioDraft(setRatioDrafts, index, patch);
  const handleStudyRatioChange = (index: number, patch: Partial<RatioDefinition>) =>
    updateRatioDraft(setStudyRatioDrafts, index, patch);
  const updateChannelDraft = (
    setter: React.Dispatch<React.SetStateAction<ChannelDefinition[]>>,
    channel: number,
    patch: Partial<ChannelDefinition>
  ) => {
    setter((prev) =>
      normalizeChannelDefinitions(
        prev.map((entry) => {
          if (entry.channel !== channel) {
            return entry;
          }
          return {
            ...entry,
            ...patch,
            channel: entry.channel
          };
        }),
        prev.map((entry) => entry.channel)
      )
    );
  };
  const handleConfigChannelChange = (channel: number, patch: Partial<ChannelDefinition>) =>
    updateChannelDraft(setChannelDrafts, channel, patch);
  const handleStudyChannelChange = (channel: number, patch: Partial<ChannelDefinition>) =>
    updateChannelDraft(setStudyChannelDrafts, channel, patch);
  const addConfigRatio = () => addRatioDraft(setRatioDrafts);
  const addStudyRatio = () => addRatioDraft(setStudyRatioDrafts);
  const removeConfigRatio = (index: number) => removeRatioDraft(setRatioDrafts, index);
  const removeStudyRatio = (index: number) => removeRatioDraft(setStudyRatioDrafts, index);
  const resetConfigRatios = () => setRatioDrafts(DEFAULT_RATIO_DEFINITIONS);
  const resetConfigChannels = () => setChannelDrafts(buildDefaultChannelDefinitions(configChannels.map((channel) => channel.channel)));
  const resetStudyRatios = () => setStudyRatioDrafts(ratioDefinitions);
  const resetStudyChannels = () => setStudyChannelDrafts(normalizeChannelDefinitions(channelDefinitions));
  const handleStudyRatioSave = async () => {
    if (!study) return;
    try {
      const response = await updateRatiosMutation.mutateAsync(studyRatioDrafts);
      setRatioDefinitions(response.ratios);
    } catch (error) {
      /* handled below */
    }
  };
  const handleStudyChannelSave = async () => {
    if (!study) return;
    try {
      const response = await updateChannelsMutation.mutateAsync(studyChannelDrafts);
      const normalized = normalizeChannelDefinitions(response.channels, studyChannels.map((channel) => channel.channel));
      setChannelDefinitions(normalized);
      updateStudy({ channel_definitions: normalized });
    } catch (error) {
      /* handled below */
    }
  };

  const configChannelLabelMap = useMemo(
    () =>
      normalizeChannelDefinitions(channelDrafts, channelDrafts.map((definition) => definition.channel)).reduce<Record<number, string>>((acc, definition) => {
        acc[definition.channel] = definition.label;
        return acc;
      }, {}),
    [channelDrafts]
  );
  const studyChannelLabelMap = useMemo(
    () =>
      normalizeChannelDefinitions(studyChannelDrafts, studyChannelDrafts.map((definition) => definition.channel)).reduce<Record<number, string>>((acc, definition) => {
        acc[definition.channel] = definition.label;
        return acc;
      }, {}),
    [studyChannelDrafts]
  );

  useEffect(() => {
    setChannelWindowDrafts(buildChannelWindowDrafts(previewChannelRanges, thresholds));
  }, [
    previewChannelRanges,
    thresholds,
  ]);

  const buildSliderChannelWindowDraft = (
    channel: ChannelKey,
    value: number[],
    activeThumb: number
  ): ChannelWindowDraft => {
    const [rangeMin, rangeMax] = previewChannelRanges[channel] ?? [0, getChannelLimit(channel)];
    const limit = getChannelLimit(channel);
    const draft = channelWindowDrafts[channel] ?? {
      min: String(rangeMin),
      threshold: String(thresholds[channel] ?? 0),
      max: String(rangeMax)
    };
    const currentMin = Math.min(parseIntensityDraft(draft.min) ?? rangeMin, limit);
    const currentThreshold = Math.min(parseIntensityDraft(draft.threshold) ?? thresholds[channel] ?? 0, limit);
    const currentMax = Math.min(parseIntensityDraft(draft.max) ?? rangeMax, limit);

    if (activeThumb === 0) {
      return {
        min: String(clampIntensity(Math.min(value[0], currentThreshold), limit)),
        threshold: String(clampIntensity(currentThreshold, limit)),
        max: String(clampIntensity(currentMax, limit)),
      };
    }

    if (activeThumb === 2) {
      return {
        min: String(clampIntensity(currentMin, limit)),
        threshold: String(clampIntensity(currentThreshold, limit)),
        max: String(clampIntensity(Math.max(value[2], currentThreshold), limit)),
      };
    }

    const nextThreshold = clampIntensity(value[1], limit);
    return {
      min: String(clampIntensity(Math.min(currentMin, nextThreshold), limit)),
      threshold: String(nextThreshold),
      max: String(clampIntensity(Math.max(currentMax, nextThreshold), limit)),
    };
  };

  const commitChannelWindow = (
    channel: ChannelKey,
    _source: ChannelWindowCommitSource,
    overrideDraft?: ChannelWindowDraft
  ) => {
    const [currentMin, currentMax] = previewChannelRanges[channel] ?? [0, getChannelLimit(channel)];
    const draft = overrideDraft ?? channelWindowDrafts[channel] ?? {
      min: String(currentMin),
      threshold: String(thresholds[channel] ?? 0),
      max: String(currentMax)
    };
    const limit = getChannelLimit(channel);
    const minCandidate = Math.min(parseIntensityDraft(draft.min) ?? currentMin, limit);
    const thresholdCandidate = Math.min(parseIntensityDraft(draft.threshold) ?? thresholds[channel] ?? 0, limit);
    const maxCandidate = Math.min(parseIntensityDraft(draft.max) ?? currentMax, limit);

    // Histogram/statistics should follow threshold only; Min/Max are display bounds.
    const resolvedThreshold = thresholdCandidate;
    const resolvedMin = Math.min(minCandidate, thresholdCandidate);
    const resolvedMax = Math.max(maxCandidate, thresholdCandidate);

    const normalizedMin = clampIntensity(resolvedMin, limit);
    const normalizedThreshold = clampIntensity(resolvedThreshold, limit);
    const normalizedMax = clampIntensity(resolvedMax, limit);

    if (currentMin !== normalizedMin || currentMax !== normalizedMax) {
      setPreviewChannelRange(channel, [normalizedMin, normalizedMax]);
    }
    if (thresholds[channel] !== normalizedThreshold) {
      setThreshold(channel, normalizedThreshold);
    }
    setChannelWindowDrafts((previous) => ({
      ...previous,
      [channel]: {
        min: String(normalizedMin),
        threshold: String(normalizedThreshold),
        max: String(normalizedMax),
      },
    }));
  };

  const handleChannelWindowDraftChange = (channel: ChannelKey, field: keyof ChannelWindowDraft, value: string) => {
    setChannelWindowDrafts((previous) => ({
      ...previous,
      [channel]: {
        ...(previous[channel] ?? {
          min: String(previewChannelRanges[channel]?.[0] ?? 0),
          threshold: String(thresholds[channel] ?? 0),
          max: String(previewChannelRanges[channel]?.[1] ?? getChannelLimit(channel))
        }),
        [field]: value,
      },
    }));
  };

  const handleChannelWindowKeyDown = (
    event: React.KeyboardEvent<HTMLElement>,
    channel: ChannelKey,
    source: Exclude<ChannelWindowCommitSource, "slider">
  ) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commitChannelWindow(channel, source);
      event.currentTarget.blur();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setChannelWindowDrafts((previous) => ({
        ...previous,
        [channel]: {
          min: String(previewChannelRanges[channel]?.[0] ?? 0),
          threshold: String(thresholds[channel] ?? 0),
          max: String(previewChannelRanges[channel]?.[1] ?? 0),
        },
      }));
      event.currentTarget.blur();
    }
  };

  const removeRatioDraft = (setter: React.Dispatch<React.SetStateAction<RatioDefinition[]>>, index: number) => {
    setter((prev) => prev.filter((_, idx) => idx !== index));
  };

  const availableGroupNames = useMemo(() => Object.keys(groupMap).sort(), [groupMap]);
  const configChannels = useMemo(
    () => normalizeChannelDefinitions(channelDrafts, channelDrafts.map((definition) => definition.channel)),
    [channelDrafts]
  );
  const studyChannels = useMemo(
    () => normalizeChannelDefinitions(studyChannelDrafts, studyChannelDrafts.map((definition) => definition.channel)),
    [studyChannelDrafts]
  );
  const activeStudyChannels = useMemo(
    () => normalizeChannelDefinitions(channelDefinitions, channelDefinitions.map((definition) => definition.channel)),
    [channelDefinitions]
  );
  const hasGroups = availableGroupNames.length > 0;
  const builderSummary = useMemo(() => {
    const subjectCount = Object.values(groupMap).reduce((acc, subjects) => acc + subjects.length, 0);
    return { groupCount: availableGroupNames.length, subjectCount };
  }, [availableGroupNames, groupMap]);

  useEffect(() => {
    if (!statusQuery.data) {
      return;
    }
    if (statusQuery.data.state === "succeeded" && statusQuery.data.output_path) {
      setResultsPath(statusQuery.data.output_path);
    }
  }, [statusQuery.data]);

  const scanSummary = useMemo(() => {
    if (!scanResult) return null;
    const groupCount = scanResult.groups.length;
    const subjectCount = scanResult.groups.reduce((acc, group) => acc + group.subjects.length, 0);
    return { groupCount, subjectCount, files: scanResult.nd2_files.length };
  }, [scanResult]);

  const cacheInfo = useMemo(() => {
    if (!statusQuery.data || !statusQuery.data.latest_source_mtime) {
      return null;
    }
    const timestamp = new Date(statusQuery.data.latest_source_mtime);
    return {
      timestamp: timestamp.toLocaleString(),
      hash: statusQuery.data.source_hash ? statusQuery.data.source_hash.slice(0, 10) : null
    };
  }, [statusQuery.data]);

  const browseForFile = (inputRef: React.RefObject<HTMLInputElement>) => {
    inputRef.current?.click();
  };

  const handleUploadSelection = async (
    event: React.ChangeEvent<HTMLInputElement>,
    category: "config" | "threshold_results"
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setUploadError(null);
      const response = await uploadMutation.mutateAsync({ category, file });
      if (category === "config") {
        setConfigPath(response.stored_path);
        setConfigOriginalName(response.original_name || null);
        setConfigError(null);
      } else {
        setResultsPath(response.stored_path);
        setResultsOriginalName(response.original_name || null);
      }
    } catch (error) {
      const message = getErrorMessage(error);
      if (category === "config") {
        setConfigError(message);
      } else {
        setUploadError(message);
      }
    } finally {
      event.target.value = "";
    }
  };

  useEffect(() => {
    if (study?.nd2_root && !inputDir) {
      setInputDir(study.nd2_root);
    }
  }, [study?.nd2_root, inputDir]);

  const workflowSteps = useMemo<Array<{ id: StepId; title: string; description: string; completed: boolean; state: GuideState }>>(() => {
    const projectComplete = Boolean(scanResult);
    const runComplete = Boolean(resultsPath) || Boolean(statusQuery.data?.state === "succeeded" && statusQuery.data?.output_path);
    const configurationComplete = Boolean(configPath && hasGroups && runComplete);
    const analysisComplete = Boolean(study);
    const base: Array<{ id: StepId; title: string; description: string; completed: boolean }> = [
      {
        id: "project",
        title: "Project",
        description: "Scan microscopy data",
        completed: projectComplete
      },
      {
        id: "configuration",
        title: "Configuration",
        description: "Set groups + generate",
        completed: configurationComplete
      },
      {
        id: "analysis",
        title: "Analysis",
        description: "Load study + inspect",
        completed: analysisComplete
      }
    ];
    const firstPending = base.find((step) => !step.completed)?.id ?? null;
    return base.map((step) => ({
      ...step,
      state: step.completed ? "completed" : step.id === firstPending ? "active" : "upcoming"
    }));
  }, [
    configPath,
    hasGroups,
    scanResult,
    resultsPath,
    statusQuery.data?.output_path,
    statusQuery.data?.state,
    study
  ]);

  const stepStateById = useMemo(
    () =>
      workflowSteps.reduce<Record<StepId, GuideState>>((acc, step) => {
        acc[step.id] = step.state;
        return acc;
      }, {} as Record<StepId, GuideState>),
    [workflowSteps]
  );

  const getStepState = (stepId: StepId): GuideState => stepStateById[stepId] ?? "upcoming";
  const sectionSx = (stepId: StepId) => sectionStylesForState(getStepState(stepId));
  const toggleModule = (moduleId: ModuleId) => {
    setCollapsedModules((previous) => ({
      ...previous,
      [moduleId]: !previous[moduleId],
    }));
  };
  const renderModule = (
    moduleId: ModuleId,
    stepId: StepId,
    title: string,
    children: React.ReactNode
  ) => (
    <Box sx={sectionSx(stepId)}>
      <Stack spacing={1.25}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Typography variant="subtitle1">{title}</Typography>
          <IconButton size="small" onClick={() => toggleModule(moduleId)}>
            {collapsedModules[moduleId] ? <ExpandMoreIcon fontSize="small" /> : <ExpandLessIcon fontSize="small" />}
          </IconButton>
        </Stack>
        <Collapse in={!collapsedModules[moduleId]}>
          <Box>{children}</Box>
        </Collapse>
      </Stack>
    </Box>
  );
  const showProjectModules = activeStage === "project";
  const showConfigurationModules = activeStage === "configuration";
  const showAnalysisModules = activeStage === "analysis";

  return (
    <Stack spacing={2} px={3} py={3} sx={{ minHeight: "100%" }}>
      {!study && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Study Pipeline
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Scan microscopy files, configure groups and modules, then run threshold generation and analysis.
          </Typography>
        </Box>
      )}

      <Box
        sx={{
          borderRadius: 2,
          border: "1px solid rgba(15,23,42,0.1)",
          p: 2,
          backgroundColor: "#ffffff"
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          Workflow stages
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
          Select a stage to focus only on its related modules.
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
            gap: 1
          }}
        >
          {workflowSteps.map((step, index) => {
            const isSelected = activeStage === step.id;
            return (
              <Box
                key={step.id}
                onClick={() => setActiveStage(step.id)}
                sx={{
                  borderRadius: 1.25,
                  p: 0.9,
                  minHeight: 78,
                  border: `1px solid ${
                    isSelected ? "rgba(37,99,235,0.6)" : step.state === "completed" ? "rgba(34,197,94,0.45)" : "rgba(15,23,42,0.1)"
                  }`,
                  backgroundColor: isSelected ? "rgba(37,99,235,0.1)" : "#fff",
                  cursor: "pointer"
                }}
              >
                <Stack direction="row" spacing={0.5} alignItems="center">
                  <Chip
                    label={index + 1}
                    size="small"
                    color={isSelected ? "primary" : step.state === "completed" ? "success" : "default"}
                    variant={isSelected || step.state === "completed" ? "filled" : "outlined"}
                    sx={{ minWidth: 30 }}
                  />
                <Typography variant="caption" sx={{ fontWeight: 600, lineHeight: 1.15 }}>
                  {step.title}
                </Typography>
                </Stack>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5, lineHeight: 1.15, minHeight: 26 }}>
                  {step.description}
                </Typography>
              </Box>
            );
          })}
        </Box>
      </Box>

      {showProjectModules &&
        renderModule(
          "scan",
          "project",
          "Scan Directory",
          <Stack spacing={1.5}>
          <TextField
            label="Microscopy Input Directory"
            value={inputDir}
            onChange={(event) => setInputDir(event.target.value)}
            size="small"
            fullWidth
            placeholder="/path/to/images"
          />
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              size="small"
              disabled={!inputDir || scanMutation.isPending}
              onClick={async () => {
                try {
                  const response = await scanMutation.mutateAsync({
                    input_dir: inputDir,
                    subject_strategy: scanSubjectStrategy
                  });
                  setScanResult(response);
                  const groups: Record<string, string[]> = {};
                  response.groups.forEach((group) => {
                    groups[group.group_name] = group.subjects.map((subject) => subject.subject_id);
                  });
                  const normalized = normalizeGroupMapping(groups);
                  setGroupMap(normalized);
                  setGroupsJson(JSON.stringify(normalized, null, 2));
                  setChannelDrafts(
                    normalizeChannelDefinitions(
                      response.channel_definitions,
                      response.channel_definitions?.map((definition) => definition.channel)
                    )
                  );
                  setConfigError(null);
                  setPixelSize("");
                } catch (error) {
                  /* handled below */
                }
              }}
            >
              {scanMutation.isPending ? "Scanning..." : "Scan Directory"}
            </Button>
          </Stack>
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={scanSubjectStrategy === "auto"}
                onChange={(event) => setScanSubjectStrategy(event.target.checked ? "auto" : "per_file")}
              />
            }
            label="Auto infer shared subject IDs from filename tokens"
          />
          <Typography variant="caption" color="text.secondary">
            {builderSummary.groupCount} editable groups • {builderSummary.subjectCount} subjects in the builder.
          </Typography>
          {scanSummary && (
            <Typography variant="caption" color="text.secondary">
              Last scan detected {scanSummary.groupCount} groups across {scanSummary.subjectCount} subjects ({scanSummary.files} files: ND2/CZI/OIB/OIF).
            </Typography>
          )}
          {availableGroupNames.length > 0 && (
            <Box sx={{ maxHeight: 120, overflowY: "auto", px: 1, py: 0.5, borderRadius: 1, backgroundColor: "rgba(255,255,255,0.6)" }}>
              {availableGroupNames.map((group) => (
                <Typography key={group} variant="caption" display="block">
                  {group}: {(groupMap[group] ?? []).length} subjects
                </Typography>
              ))}
            </Box>
          )}
          {scanMutation.isError && <Alert severity="error">{getErrorMessage(scanMutation.error)}</Alert>}
        </Stack>
        )}

      {showConfigurationModules &&
        renderModule(
          "config_builder",
          "configuration",
          "Build Configuration",
          scanResult || hasGroups ? (
          <Stack spacing={1.75}>
            <Stack spacing={1}>
              <Typography variant="subtitle1">Natural Language Grouping</Typography>
              <Typography variant="caption" color="text.secondary">
                Describe how files should be grouped (for complex naming patterns). Requires `OPENAI_API_KEY` on the backend.
              </Typography>
              <TextField
                size="small"
                fullWidth
                label="Optional model"
                placeholder="gpt-4.1-mini"
                value={autoGroupModel}
                onChange={(event) => setAutoGroupModel(event.target.value)}
              />
              <TextField
                label="Grouping instructions"
                size="small"
                fullWidth
                multiline
                minRows={3}
                placeholder="Example: Group all subject IDs ending with odd numbers into Treatment A and even numbers into Control."
                value={autoGroupInstructions}
                onChange={(event) => setAutoGroupInstructions(event.target.value)}
              />
              <Stack direction="row" spacing={1} alignItems="center">
                <Button
                  variant="outlined"
                  size="small"
                  disabled={!inputDir || !autoGroupInstructions.trim() || autoGroupsMutation.isPending}
                  onClick={async () => {
                    try {
                      const response = await autoGroupsMutation.mutateAsync({
                        input_dir: inputDir,
                        instructions: autoGroupInstructions.trim(),
                        model: autoGroupModel.trim() || undefined
                      });
                      const normalized = normalizeGroupMapping(response.groups);
                      setGroupMap(normalized);
                      setGroupsJson(JSON.stringify(normalized, null, 2));
                      setSubjectInputs({});
                      setConfigError(null);
                    } catch (error) {
                      setConfigError(getErrorMessage(error));
                    }
                  }}
                >
                  {autoGroupsMutation.isPending ? "Generating..." : "Generate groups with AI"}
                </Button>
                {autoGroupsMutation.isSuccess && (
                  <Typography variant="caption" color="text.secondary">
                    Generated with {autoGroupsMutation.data.model}
                  </Typography>
                )}
              </Stack>
              {autoGroupsMutation.isError && (
                <Alert severity="error">{getErrorMessage(autoGroupsMutation.error)}</Alert>
              )}
              {autoGroupsMutation.data?.notes && (
                <Alert severity="info">{autoGroupsMutation.data.notes}</Alert>
              )}
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Button
                variant="outlined"
                size="small"
                disabled={!hasGroups || configMutation.isPending || !scanResult}
                onClick={async () => {
                  if (!scanResult) return;
                  if (!hasGroups) {
                    setConfigError("Add at least one group with subjects before creating a config.");
                    return;
                  }
                  let parsedGroups: Record<string, string[]>;
                  try {
                    if (Object.keys(groupMap).length > 0) {
                      parsedGroups = normalizeGroupMapping(groupMap);
                    } else {
                      parsedGroups = JSON.parse(groupsJson || "{}");
                      if (typeof parsedGroups !== "object" || parsedGroups === null || Array.isArray(parsedGroups)) {
                        throw new Error("Groups must map group names to arrays of subject IDs.");
                      }
                      parsedGroups = normalizeGroupMapping(parsedGroups);
                    }
                    setConfigError(null);
                  } catch (error) {
                    setConfigError(getErrorMessage(error));
                    return;
                  }
                  try {
                    const response = await configMutation.mutateAsync({
                      input_dir: scanResult.input_dir,
                      study_name: scanResult.study_name,
                      groups: parsedGroups,
                      pixel_size_um: pixelSize ? Number(pixelSize) : undefined,
                      output_path: configPath || undefined,
                      ratios: ratioDrafts,
                      channel_definitions: configChannels
                    });
                    setConfigPath(response.config_path);
                    setGroupsJson(JSON.stringify(parsedGroups, null, 2));
                    setGroupMap(parsedGroups);
                    setSubjectInputs({});
                    setConfigError(null);
                  } catch (error) {
                    /* handled below */
                  }
                }}
              >
                {configMutation.isPending ? "Saving..." : "Create Config"}
              </Button>
              <Typography variant="caption" color="text.secondary">
                Save a config JSON once the group assignments look correct.
              </Typography>
            </Stack>
            {configMutation.isError && <Alert severity="error">{getErrorMessage(configMutation.error)}</Alert>}
            {configError && <Alert severity="error">{configError}</Alert>}
            {scannedSubjects.length > 0 && (
              <Stack spacing={1.25}>
                <Typography variant="subtitle1">Subject & Replica Assignment</Typography>
                <Typography variant="caption" color="text.secondary">
                  Assign detected subjects to groups and verify which replica files will be included.
                </Typography>
                <Box
                  sx={{
                    maxHeight: 300,
                    overflowY: "auto",
                    borderRadius: 1.5,
                    border: "1px solid rgba(15,23,42,0.08)",
                    p: 1.25,
                    backgroundColor: "rgba(15,23,42,0.02)"
                  }}
                >
                  <Stack spacing={1}>
                    {scannedSubjects.map((subject) => {
                      const assignedGroup = assignedGroupBySubject[subject.subjectId] ?? subject.detectedGroup;
                      const groupChoices = Array.from(
                        new Set([subject.detectedGroup, assignedGroup, ...availableGroupNames].filter(Boolean))
                      ).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
                      const previewFiles = subject.replicates.slice(0, 2).join(", ");
                      const remainingFiles = Math.max(0, subject.replicates.length - 2);

                      return (
                        <Box
                          key={`subject-assignment-${subject.subjectId}`}
                          sx={{
                            borderRadius: 1,
                            border: "1px solid rgba(15,23,42,0.08)",
                            backgroundColor: "#fff",
                            p: 1
                          }}
                        >
                          <Stack
                            direction={{ xs: "column", sm: "row" }}
                            spacing={1}
                            alignItems={{ sm: "center" }}
                            justifyContent="space-between"
                          >
                            <Typography variant="body2" sx={{ minWidth: 110, fontWeight: 600 }}>
                              {subject.subjectId}
                            </Typography>
                            <TextField
                              select
                              size="small"
                              label="Group"
                              value={assignedGroup}
                              onChange={(event) => assignSubjectToGroup(subject.subjectId, event.target.value as string)}
                              sx={{ minWidth: 180 }}
                            >
                              {groupChoices.map((groupName) => (
                                <MenuItem key={`subject-${subject.subjectId}-${groupName}`} value={groupName}>
                                  {groupName}
                                </MenuItem>
                              ))}
                            </TextField>
                            <Typography variant="caption" color="text.secondary">
                              {subject.replicateCount} replicas
                            </Typography>
                          </Stack>
                          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                            {previewFiles || "No replica files detected"}
                            {remainingFiles > 0 ? ` +${remainingFiles} more` : ""}
                          </Typography>
                        </Box>
                      );
                    })}
                  </Stack>
                </Box>
              </Stack>
            )}
            <Stack spacing={1.75}>
              <Typography variant="subtitle1">Group Builder</Typography>
              <Typography variant="caption" color="text.secondary">
                Add each treatment group and list the subject IDs that belong to it. You can still fall back to the generated JSON if you prefer.
              </Typography>
              <Stack direction="row" spacing={1}>
                <TextField
                  label="New group name"
                  size="small"
                  fullWidth
                  value={newGroupName}
                  onChange={(event) => setNewGroupName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      handleAddGroup();
                    }
                  }}
                />
                <Button variant="outlined" size="small" onClick={handleAddGroup} disabled={!newGroupName.trim()}>
                  Add Group
                </Button>
              </Stack>
              {availableGroupNames.length === 0 ? (
                <Typography variant="caption" color="text.secondary">
                  No groups yet. Add a group above, then assign subjects to it.
                </Typography>
              ) : (
                <Stack spacing={1.25}>
                  {availableGroupNames.map((groupName) => {
                    const selectedSubjects = groupMap[groupName] ?? [];
                    const suggestions = scanDefaults[groupName] ?? [];
                    const pendingSubject = subjectInputs[groupName] ?? "";
                    return (
                      <Box
                        key={groupName}
                        sx={{
                          borderRadius: 1,
                          border: "1px solid rgba(15,23,42,0.08)",
                          backgroundColor: "rgba(15,23,42,0.02)",
                          p: 1.5
                        }}
                      >
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">{groupName}</Typography>
                          <Button variant="text" size="small" color="error" onClick={() => handleRemoveGroup(groupName)}>
                            Remove group
                          </Button>
                        </Stack>
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
                          <TextField
                            label="Add subject ID"
                            size="small"
                            value={pendingSubject}
                            onChange={(event) =>
                              setSubjectInputs((prev) => ({
                                ...prev,
                                [groupName]: event.target.value
                              }))
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter") {
                                event.preventDefault();
                                handleAddSubject(groupName, pendingSubject);
                              }
                            }}
                          />
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleAddSubject(groupName, pendingSubject)}
                            disabled={!pendingSubject.trim()}
                          >
                            Add subject
                          </Button>
                        </Stack>
                        {suggestions.length > 0 && (
                          <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mt: 1 }}>
                            {suggestions.map((subjectId) => (
                              <Chip
                                key={`${groupName}-${subjectId}-suggestion`}
                                label={subjectId}
                                size="small"
                                variant="outlined"
                                onClick={() => handleAddSubject(groupName, subjectId)}
                              />
                            ))}
                          </Stack>
                        )}
                        <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mt: 1 }}>
                          {selectedSubjects.length === 0 ? (
                            <Typography variant="caption" color="text.secondary">
                              No subjects assigned yet.
                            </Typography>
                          ) : (
                            selectedSubjects.map((subjectId) => (
                              <Chip
                                key={`${groupName}-${subjectId}`}
                                label={subjectId}
                                size="small"
                                onDelete={() => handleRemoveSubject(groupName, subjectId)}
                              />
                            ))
                          )}
                        </Stack>
                      </Box>
                    );
                  })}
                </Stack>
              )}
              <Stack direction="row" spacing={1}>
                <Button
                  variant="text"
                  size="small"
                  disabled={Object.keys(scanDefaults).length === 0}
                  onClick={() => {
                    const reset = Object.fromEntries(Object.entries(scanDefaults).map(([key, value]) => [key, [...value]]));
                    const normalized = normalizeGroupMapping(reset);
                    setGroupMap(normalized);
                    setGroupsJson(JSON.stringify(normalized, null, 2));
                    setSubjectInputs({});
                    setConfigError(null);
                  }}
                >
                  Reset to scanned defaults
                </Button>
                <Button
                  variant="text"
                  size="small"
                  onClick={() => {
                    try {
                      const parsed = JSON.parse(groupsJson || "{}");
                      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
                        throw new Error("Group JSON must map group names to arrays of subject IDs.");
                      }
                      const normalized = normalizeGroupMapping(parsed as Record<string, string[]>);
                      setGroupMap(normalized);
                      setGroupsJson(JSON.stringify(normalized, null, 2));
                      setSubjectInputs({});
                      setConfigError(null);
                    } catch (error) {
                      setConfigError(getErrorMessage(error));
                    }
                  }}
                >
                  Apply JSON override
                </Button>
              </Stack>
              <TextField
                label="Group JSON override"
                value={groupsJson}
                onChange={(event) => setGroupsJson(event.target.value)}
                multiline
                minRows={4}
                maxRows={12}
                size="small"
                fullWidth
                helperText="JSON mapping of group -> subjects. Edit and choose Apply JSON override to sync the builder."
              />
            </Stack>
            <Stack spacing={1.25}>
              <Stack direction="row" alignItems="center" justifyContent="space-between">
                <Typography variant="subtitle1">Config Channels</Typography>
                <Button variant="text" size="small" onClick={resetConfigChannels} disabled={configMutation.isPending}>
                  Reset
                </Button>
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Name channels and choose pseudocolors for preview rendering and plot labels.
              </Typography>
              {configChannels.map((channel) => (
                <Box
                  key={`config-channel-${channel.channel}`}
                  sx={{
                    border: "1px solid rgba(15,23,42,0.08)",
                    borderRadius: 2,
                    p: 1.25,
                    backgroundColor: "#fff"
                  }}
                >
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                    <Typography variant="caption" sx={{ minWidth: 90, fontWeight: 600 }}>
                      Ch{channel.channel}
                    </Typography>
                    <TextField
                      label="Label"
                      size="small"
                      fullWidth
                      value={channel.label}
                      onChange={(event) => handleConfigChannelChange(channel.channel, { label: event.target.value })}
                    />
                    <TextField
                      type="color"
                      label="Color"
                      size="small"
                      value={channel.color}
                      onChange={(event) => handleConfigChannelChange(channel.channel, { color: event.target.value })}
                      sx={{ width: 110 }}
                      inputProps={{ style: { padding: 0, height: 34 } }}
                    />
                  </Stack>
                </Box>
              ))}
            </Stack>
            <Stack spacing={1.25}>
              <Stack direction="row" alignItems="center" justifyContent="space-between">
                <Typography variant="subtitle1">Config Ratios</Typography>
                <Button variant="text" size="small" onClick={resetConfigRatios} disabled={configMutation.isPending}>
                  Reset
                </Button>
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Choose which channel ratios should be available when collaborators load this config.
              </Typography>
              {ratioDrafts.map((ratio, index) => (
                <Box
                  key={ratio.id ?? `config-ratio-${index}`}
                  sx={{
                    border: "1px solid rgba(15,23,42,0.08)",
                    borderRadius: 2,
                    p: 1.25,
                    backgroundColor: "#fff"
                  }}
                >
                  <Stack spacing={1}>
                    <TextField
                      label="Ratio name"
                      size="small"
                      fullWidth
                      value={ratio.label}
                      onChange={(event) => handleConfigRatioChange(index, { label: event.target.value })}
                    />
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                      <TextField
                        select
                        fullWidth
                        size="small"
                        label="Numerator channel"
                        value={ratio.numerator_channel}
                        onChange={(event) =>
                          handleConfigRatioChange(index, { numerator_channel: Number(event.target.value) })
                        }
                      >
                        {configChannels.map((channelDefinition) => (
                          <MenuItem key={`config-ratio-num-${index}-${channelDefinition.channel}`} value={channelDefinition.channel}>
                            {configChannelLabelMap[channelDefinition.channel] ?? `Channel ${channelDefinition.channel}`}
                          </MenuItem>
                        ))}
                      </TextField>
                      <TextField
                        select
                        fullWidth
                        size="small"
                        label="Denominator channel"
                        value={ratio.denominator_channel}
                        onChange={(event) =>
                          handleConfigRatioChange(index, { denominator_channel: Number(event.target.value) })
                        }
                      >
                        {configChannels.map((channelDefinition) => (
                          <MenuItem key={`config-ratio-den-${index}-${channelDefinition.channel}`} value={channelDefinition.channel}>
                            {configChannelLabelMap[channelDefinition.channel] ?? `Channel ${channelDefinition.channel}`}
                          </MenuItem>
                        ))}
                      </TextField>
                      <Button
                        variant="text"
                        size="small"
                        color="error"
                        onClick={() => removeConfigRatio(index)}
                        disabled={ratioDrafts.length <= 1}
                        sx={{ alignSelf: { xs: "flex-start", sm: "center" }, whiteSpace: "nowrap" }}
                      >
                        Remove
                      </Button>
                    </Stack>
                  </Stack>
                </Box>
              ))}
              <Button
                variant="outlined"
                size="small"
                onClick={addConfigRatio}
                disabled={ratioDrafts.length >= 6 || configMutation.isPending}
              >
                Add ratio
              </Button>
            </Stack>
            <TextField
              label="Pixel size (µm)"
              value={pixelSize}
              onChange={(event) => setPixelSize(event.target.value)}
              size="small"
              type="number"
              inputProps={{ step: "0.001" }}
            />
          </Stack>
        ) : (
          <Typography variant="caption" color="text.secondary">
            Scan a microscopy directory to unlock group editing, JSON overrides, channel presets, ratio presets, and config exports.
          </Typography>
        )
        )}

      {showAnalysisModules &&
        study &&
        renderModule(
          "analysis_controls",
          "analysis",
          "Analysis Controls",
          <Stack spacing={1.5}>
        <Typography variant="caption" color="text.secondary">
          Default mode only: positive signal fraction (%) from thresholded channels.
        </Typography>
        <Box>
          <Typography variant="caption" color="text.secondary">
            Subjects per group in preview: {previewSamplesPerGroup}
          </Typography>
          <Slider
            size="small"
            min={1}
            max={20}
            step={1}
            marks={[1, 5, 10, 15, 20].map((value) => ({ value, label: String(value) }))}
            value={previewSamplesPerGroup}
            onChange={(_, value) => {
              const numeric = Array.isArray(value) ? value[0] : value;
              setPreviewSamplesPerGroup(typeof numeric === "number" ? numeric : 1);
            }}
            sx={{ mt: 0.5, maxWidth: 280 }}
          />
          <Typography variant="caption" color="text.secondary">
            Set this above 1, then click subject chips under each group in the Preview pane to switch pictures.
          </Typography>
        </Box>
        <Typography variant="subtitle1">Threshold &amp; Range</Typography>
        <Typography variant="caption" color="text.secondary">
          Threshold only controls the mask; Min/Max control the raw intensity window for live previews. Overlay combines both.
        </Typography>
        {activeStudyChannels.map((definition) => {
          const channel = `channel_${definition.channel}`;
          const limit = getChannelLimit(channel);
          const range = previewChannelRanges[channel] ?? [0, limit];
          const thresholdValue = Math.min(Math.max(thresholds[channel] ?? 0, range[0]), range[1]);
          const draft = channelWindowDrafts[channel] ?? {
            min: String(range[0]),
            threshold: String(thresholdValue),
            max: String(range[1])
          };
          const sliderValue = sliderValueFromDraft(range, thresholdValue, draft);
          const displayMin = draft.min.trim() || String(range[0]);
          const displayThreshold = draft.threshold.trim() || String(thresholdValue);
          const displayMax = draft.max.trim() || String(range[1]);
          const channelSliderMarks = [0, 0.25, 0.5, 0.75, 1].map((fraction, index) => ({
            value: Math.round(limit * fraction),
            label: index === 0 || index === 4 ? String(Math.round(limit * fraction)) : ""
          }));
          return (
            <Box key={channel}>
              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                <Stack direction="row" spacing={0.75} alignItems="center">
                  <Box
                    sx={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      backgroundColor: definition.color,
                      border: "1px solid rgba(15,23,42,0.2)"
                    }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {definition.label}
                  </Typography>
                </Stack>
                <Typography variant="caption" color="text.secondary">
                  {`Min ${displayMin} • Thr ${displayThreshold} • Max ${displayMax}`}
                </Typography>
              </Stack>
              <Slider
                size="small"
                min={0}
                max={limit}
                marks={channelSliderMarks}
                disableSwap
                onMouseEnter={() => setThresholdControlHovered(true)}
                onMouseLeave={() => setThresholdControlHovered(false)}
                sx={{
                  color: "rgba(15,23,42,0.35)",
                  mt: 1,
                  "& .MuiSlider-thumb": {
                    width: 14,
                    height: 14,
                    boxShadow: "0 1px 2px rgba(15,23,42,0.12)"
                  },
                  // Threshold handle (middle) as a short strip
                  "& .MuiSlider-thumb[data-index='1']": {
                    bgcolor: "#64748b",
                    width: 28,
                    height: 8,
                    borderRadius: 4,
                    border: "1px solid #cbd5e1"
                  },
                  // Min & Max handles as blue round knobs
                  "& .MuiSlider-thumb[data-index='0'], & .MuiSlider-thumb[data-index='2']": {
                    bgcolor: "#2563eb",
                    border: "2px solid #bfdbfe",
                    width: 16,
                    height: 16,
                    borderRadius: 999,
                    boxShadow: "0 0 0 4px rgba(37,99,235,0.18)"
                  }
                }}
                value={sliderValue}
                onChange={(_event, value) => {
                  if (!Array.isArray(value) || value.length !== 3) {
                    return;
                  }
                  const activeThumb = getActiveSliderThumbIndex();
                  activeSliderThumbRef.current[channel] = activeThumb;
                  const nextDraft = buildSliderChannelWindowDraft(channel, value, activeThumb);
                  setChannelWindowDrafts((previous) => ({
                    ...previous,
                    [channel]: nextDraft,
                  }));
                }}
                onChangeCommitted={(_event, value) => {
                  if (!Array.isArray(value) || value.length !== 3) {
                    return;
                  }
                  const activeThumb = activeSliderThumbRef.current[channel] ?? getActiveSliderThumbIndex();
                  const nextDraft = buildSliderChannelWindowDraft(channel, value, activeThumb);
                  setChannelWindowDrafts((previous) => ({
                    ...previous,
                    [channel]: nextDraft,
                  }));
                  const source: ChannelWindowCommitSource = activeThumb === 0 ? "min" : activeThumb === 2 ? "max" : "threshold";
                  commitChannelWindow(channel, source, nextDraft);
                }}
              />
              <Stack direction="row" spacing={1}>
                <TextField
                  size="small"
                  type="number"
                  label="Min"
                  value={draft.min}
                  inputProps={{ min: 0, max: limit }}
                  onChange={(event) => handleChannelWindowDraftChange(channel, "min", event.target.value)}
                  onBlur={() => commitChannelWindow(channel, "min")}
                  onKeyDown={(event) => handleChannelWindowKeyDown(event, channel, "min")}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Threshold"
                  value={draft.threshold}
                  inputProps={{ min: 0, max: limit }}
                  onChange={(event) => handleChannelWindowDraftChange(channel, "threshold", event.target.value)}
                  onBlur={() => commitChannelWindow(channel, "threshold")}
                  onKeyDown={(event) => handleChannelWindowKeyDown(event, channel, "threshold")}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Max"
                  value={draft.max}
                  inputProps={{ min: 1, max: limit }}
                  onChange={(event) => handleChannelWindowDraftChange(channel, "max", event.target.value)}
                  onBlur={() => commitChannelWindow(channel, "max")}
                  onKeyDown={(event) => handleChannelWindowKeyDown(event, channel, "max")}
                />
              </Stack>
            </Box>
          );
        })}
        <Button variant="text" size="small" onClick={resetPreviewChannelRanges}>
          Reset ranges
        </Button>
        <FormControlLabel
          control={<Checkbox size="small" checked={statisticsEnabled} onChange={(event) => setStatisticsEnabled(event.target.checked)} />}
          label="Enable statistical analysis"
        />
        <Stack
          spacing={1.25}
          sx={{
            borderRadius: 2,
            border: "1px solid rgba(15,23,42,0.08)",
            backgroundColor: "rgba(15,23,42,0.02)",
            p: 1.5
          }}
        >
          <Typography variant="caption" color="text.secondary">
            Configure statistical comparisons
          </Typography>
          <FormControl size="small" fullWidth>
            <InputLabel id="test-type-label">Test method</InputLabel>
            <Select
              labelId="test-type-label"
              label="Test method"
              value={statisticsSettings.testType}
              onChange={(event) => setTestType(event.target.value as typeof statisticsSettings.testType)}
              disabled={statsControlsDisabled}
            >
              <MenuItem value="anova_parametric">Ordinary ANOVA (parametric)</MenuItem>
              <MenuItem value="anova_non_parametric">Non-parametric ANOVA (Kruskal–Wallis)</MenuItem>
              <MenuItem value="t_test">Pairwise t-tests</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" fullWidth>
            <InputLabel id="significance-display-label">Significance label</InputLabel>
            <Select
              labelId="significance-display-label"
              label="Significance label"
              value={statisticsSettings.significanceDisplay}
              onChange={(event) => setSignificanceDisplay(event.target.value as typeof statisticsSettings.significanceDisplay)}
              disabled={statsControlsDisabled}
            >
              <MenuItem value="stars">Asterisks (GraphPad style)</MenuItem>
              <MenuItem value="p_values">Exact p-values</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" fullWidth>
            <InputLabel id="comparison-mode-label">Comparison mode</InputLabel>
            <Select
              labelId="comparison-mode-label"
              label="Comparison mode"
              value={statisticsSettings.comparisonMode}
              onChange={(event) => setComparisonMode(event.target.value as typeof statisticsSettings.comparisonMode)}
              disabled={statsControlsDisabled}
            >
              <MenuItem value="all_vs_one">Reference vs. others (ANOVA + post hoc)</MenuItem>
              <MenuItem value="all_pairs">All group pairs</MenuItem>
              <MenuItem value="pairs">Custom group pairs</MenuItem>
            </Select>
          </FormControl>
          <Stack spacing={1}>
            {statisticsSettings.comparisonMode === "all_vs_one" && (
              <TextField
                select
                size="small"
                label="Reference group"
                value={statisticsSettings.referenceGroup ?? ""}
                onChange={(event) =>
                  setReferenceGroup(event.target.value ? (event.target.value as string) : null)
                }
                disabled={
                  statsControlsDisabled || statisticsSettings.comparisonMode !== "all_vs_one"
                }
                InputLabelProps={{ shrink: true }}
                SelectProps={{
                  displayEmpty: true,
                  renderValue: (selected) => (selected ? String(selected) : "Auto (first group)")
                }}
                fullWidth
              >
                <MenuItem value="">
                  <em>Auto (first group)</em>
                </MenuItem>
                {studyGroups.map((group) => (
                  <MenuItem key={group} value={group}>
                    {group}
                  </MenuItem>
                ))}
              </TextField>
            )}
            {statisticsSettings.comparisonMode === "pairs" && (
              <Stack spacing={1}>
                <Typography variant="caption" color="text.secondary">
                  Select one or more pair comparisons.
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Button
                    variant="outlined"
                    size="small"
                    disabled={statsControlsDisabled || pairOptions.length === 0}
                    onClick={() => {
                      pairOptions.forEach((pair) => addComparisonPair(pair));
                    }}
                  >
                    Select all
                  </Button>
                  {statisticsSettings.comparisonPairs.length > 0 && (
                    <Button
                      variant="text"
                      size="small"
                      onClick={clearComparisonPairs}
                      disabled={!statisticsEnabled}
                    >
                      Clear all
                    </Button>
                  )}
                </Stack>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {pairOptions.map((pair) => {
                    const key = pairToken(pair[0], pair[1]);
                    const selected = selectedPairTokens.has(key);
                    return (
                      <Chip
                        key={key}
                        clickable
                        disabled={statsControlsDisabled}
                        size="small"
                        label={`${pair[0]} ↔ ${pair[1]}`}
                        color={selected ? "primary" : "default"}
                        variant={selected ? "filled" : "outlined"}
                        onClick={() => togglePairSelection(pair)}
                      />
                    );
                  })}
                  {pairOptions.length === 0 && (
                    <Typography variant="caption" color="text.secondary">
                      At least two groups are required for pairwise comparisons.
                    </Typography>
                  )}
                </Stack>
              </Stack>
            )}
            {statisticsSettings.comparisonMode === "all_pairs" && (
              <Typography variant="caption" color="text.secondary">
                All group combinations will be compared (Tukey-style post hoc).
              </Typography>
            )}
          </Stack>
        </Stack>
      </Stack>
        )}

      {showAnalysisModules &&
        study &&
        renderModule(
          "visualization_settings",
          "analysis",
          "Visualization Settings",
          <Stack spacing={1.5}>
        <Typography variant="caption" color="text.secondary">
          Tune the figure aesthetics before exporting charts.
        </Typography>
        <TextField
          label="Chart title"
          size="small"
          value={plotSettings.title}
          onChange={(event) => setPlotTitle(event.target.value)}
          placeholder="Optional summary title"
          disabled={!study}
        />
        <Box>
          <Typography variant="caption" color="text.secondary">
            Font size: {plotSettings.fontSize}px
          </Typography>
          <Slider
            size="small"
            min={10}
            max={24}
            step={1}
            value={plotSettings.fontSize}
            onChange={(_, value) => {
              const numeric = Array.isArray(value) ? value[0] : value;
              setPlotFontSize(typeof numeric === "number" ? numeric : 13);
            }}
            sx={{ mt: 0.5, maxWidth: 240 }}
          />
        </Box>
        <FormControlLabel
          control={<Switch size="small" checked={jitterEnabled} onChange={(_, checked) => setJitterEnabled(checked)} />}
          label="Jitter data points"
        />
        {jitterEnabled && (
          <Box>
            <Typography variant="caption" color="text.secondary">
              Jitter width: {plotSettings.jitterWidth.toFixed(2)}
            </Typography>
            <Slider
              size="small"
              min={0}
              max={0.4}
              step={0.01}
              value={plotSettings.jitterWidth}
              onChange={(_, value) => {
                const numeric = Array.isArray(value) ? value[0] : value;
                setJitterWidth(typeof numeric === "number" ? numeric : 0.12);
              }}
              sx={{ mt: 0.5, maxWidth: 240 }}
            />
          </Box>
        )}
        <Stack spacing={0.75}>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="caption" color="text.secondary">
              Group palette
            </Typography>
            {studyGroups.length > 0 && (
              <Button
                variant="text"
                size="small"
                onClick={() => {
                  const reset = studyGroups.reduce<Record<string, string>>((acc, group, index) => {
                    acc[group] = defaultPalette[index % defaultPalette.length];
                    return acc;
                  }, {});
                  setPalette(reset);
                  setPalettePresetOverride(null);
                }}
              >
                Reset colors
              </Button>
            )}
          </Stack>
          {studyGroups.length === 0 && (
            <Typography variant="caption" color="text.secondary">
              Load a study to edit group colors.
            </Typography>
          )}
          {studyGroups.length > 0 && (
            <FormControl size="small" fullWidth>
              <InputLabel id="palette-preset-label">Palette preset</InputLabel>
              <Select
                labelId="palette-preset-label"
                value={palettePreset}
                label="Palette preset"
                onChange={(event) => handlePalettePresetChange(event.target.value as string)}
              >
                <MenuItem value="custom">Custom (manual)</MenuItem>
                {palettePresets.map((preset) => (
                  <MenuItem key={preset.id} value={preset.id}>
                    {preset.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {activePalettePreset && palettePreset !== "custom" && (
            <Typography variant="caption" color="text.secondary">
              {activePalettePreset.description}
            </Typography>
          )}
          {palettePreset !== "custom" && (
            <Typography variant="caption" color="text.secondary">
              Switch to Custom to fine-tune individual colors.
            </Typography>
          )}
          {studyGroups.map((group) => (
            <Stack direction="row" spacing={1} alignItems="center" key={`palette-${group}`}>
              <Typography variant="caption" sx={{ minWidth: 120 }}>
                {group}
              </Typography>
              <TextField
                type="color"
                size="small"
                value={effectivePalette[group]}
                onChange={(event) => handleGroupColorChange(group, event.target.value)}
                sx={{ width: 72 }}
                inputProps={{ style: { padding: 0, height: 32 } }}
                disabled={palettePreset !== "custom"}
              />
            </Stack>
          ))}
        </Stack>
      </Stack>
        )}

      {showConfigurationModules &&
        renderModule(
          "threshold_generation",
          "configuration",
          "Threshold Generation",
          <Stack spacing={1.5}>
        <Stack spacing={0.5}>
          <Stack direction="row" spacing={1}>
            <TextField
              label="Config Path"
              value={configPath}
              onChange={(event) => setConfigPath(event.target.value)}
              size="small"
              fullWidth
              placeholder="/path/to/config.json"
            />
            <Button
              variant="outlined"
              size="small"
              onClick={() => browseForFile(configInputRef)}
              disabled={uploadMutation.isPending}
            >
              {uploadMutation.isPending ? "Uploading..." : "Browse"}
            </Button>
          </Stack>
          {configOriginalName && (
            <Typography variant="caption" color="text.secondary">
              Uploaded from {configOriginalName}
            </Typography>
          )}
        </Stack>
        <input
          ref={configInputRef}
          type="file"
          hidden
          accept=".json,.txt,.ndjson"
          onChange={(event) => handleUploadSelection(event, "config")}
        />
        <Stack direction="row" spacing={1}>
          <Button
            variant="text"
            size="small"
            disabled={!configPath || configReadMutation.isPending}
            onClick={async () => {
              try {
                const response = await configReadMutation.mutateAsync({ path: configPath });
                const normalizedGroups = normalizeGroupMapping(response.groups);
                setGroupMap(normalizedGroups);
                setGroupsJson(JSON.stringify(normalizedGroups, null, 2));
                setSubjectInputs({});
                setPixelSize(response.pixel_size_um ? String(response.pixel_size_um) : "");
                if (response.ratios && response.ratios.length > 0) {
                  setRatioDrafts(response.ratios);
                } else {
                  setRatioDrafts(DEFAULT_RATIO_DEFINITIONS);
                }
                setChannelDrafts(
                  normalizeChannelDefinitions(
                    response.channel_definitions,
                    response.channel_definitions?.map((definition) => definition.channel)
                  )
                );
                setConfigError(null);
              } catch (error) {
                setConfigError(getErrorMessage(error));
              }
            }}
          >
            {configReadMutation.isPending ? "Loading config..." : "Load Config"}
          </Button>
        </Stack>
        <TextField
          label="Threshold Results Output"
          value={outputPath}
          onChange={(event) => setOutputPath(event.target.value)}
          size="small"
          fullWidth
          placeholder="Optional custom output path"
        />
        <Button
          variant="contained"
          size="small"
          disabled={!inputDir || !configPath || runMutation.isPending}
          onClick={async () => {
            try {
              const response = await runMutation.mutateAsync({
                input_dir: inputDir,
                config_path: configPath,
                output_path: outputPath || undefined
              });
              setJobId(response.job_id);
            } catch (error) {
              /* handled below */
            }
          }}
        >
          {runMutation.isPending ? "Launching..." : "Run Threshold Generation (All stacks)"}
        </Button>
        {statusQuery.data && (
          <Alert severity={statusQuery.data.state === "failed" ? "error" : statusQuery.data.state === "succeeded" ? "success" : "info"}>
            {statusQuery.data.state.toUpperCase()}: {statusQuery.data.message ?? "Processing"}
          </Alert>
        )}
        {statusQuery.data && (statusQuery.data.state === "queued" || statusQuery.data.state === "running") && (
          <Stack spacing={0.5}>
            <LinearProgress
              variant={statusQuery.data.progress_total ? "determinate" : "indeterminate"}
              value={
                statusQuery.data.progress_total
                  ? Math.min(
                      100,
                      (statusQuery.data.progress_completed / Math.max(statusQuery.data.progress_total, 1)) * 100
                    )
                  : undefined
              }
            />
            <Typography variant="caption" color="text.secondary">
              {statusQuery.data.progress_total
                ? `Processed ${statusQuery.data.progress_completed}/${statusQuery.data.progress_total} files`
                : "Preparing threshold generation..."}
            </Typography>
          </Stack>
        )}
        {cacheInfo && (
          <Typography variant="caption" color="text.secondary">
            Cached thresholds last verified {cacheInfo.timestamp}
            {cacheInfo.hash ? ` • fingerprint ${cacheInfo.hash}` : ""}
          </Typography>
        )}
        {runMutation.isError && <Alert severity="error">{getErrorMessage(runMutation.error)}</Alert>}
        {configReadMutation.isError && <Alert severity="error">{getErrorMessage(configReadMutation.error)}</Alert>}
      </Stack>
        )}

      {showAnalysisModules &&
        renderModule(
          "study_loader",
          "analysis",
          "Load Study Results",
          <Stack spacing={1.5}>
        <Stack spacing={0.5}>
          <Stack direction="row" spacing={1}>
            <TextField
              label="Threshold Results JSON"
              value={resultsPath}
              onChange={(event) => setResultsPath(event.target.value)}
              size="small"
              fullWidth
              placeholder="/path/to/threshold_results.json"
            />
            <Button
              variant="outlined"
              size="small"
              onClick={() => browseForFile(resultsInputRef)}
              disabled={uploadMutation.isPending}
            >
              {uploadMutation.isPending ? "Uploading..." : "Browse"}
            </Button>
          </Stack>
          {resultsOriginalName && (
            <Typography variant="caption" color="text.secondary">
              Uploaded from {resultsOriginalName}
            </Typography>
          )}
        </Stack>
        <input
          ref={resultsInputRef}
          type="file"
          hidden
          accept=".json"
          onChange={(event) => handleUploadSelection(event, "threshold_results")}
        />
        <Button
          variant="contained"
          size="small"
          disabled={!resultsPath || loadMutation.isPending}
          onClick={async () => {
            try {
              const response = await loadMutation.mutateAsync({
                file_path: resultsPath,
                input_dir_override: inputDir || undefined
              });
              setStudy(response);
            } catch (error) {
              /* handled below */
            }
          }}
        >
          {loadMutation.isPending ? "Loading..." : "Load Study"}
        </Button>
        {uploadError && <Alert severity="error">{uploadError}</Alert>}
        {loadMutation.isError && <Alert severity="error">{getErrorMessage(loadMutation.error)}</Alert>}
        {study && (
          <Alert severity="success">
            Loaded {study.study_name} • {study.groups.length} groups • {study.mice_count} subjects
          </Alert>
        )}
        {study && !study.nd2_available && (
          <Alert severity="warning">
            Source image directory unavailable at {study.nd2_root}. Mount or copy the folder, update “Microscopy Input Directory”, and load the study
            again to unlock previews.
          </Alert>
        )}
        {study && (
          <Stack spacing={0.5}>
            <Typography variant="subtitle1">Pixel size (µm/pixel)</Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
              <TextField
                label="Pixel size"
                size="small"
                type="number"
                inputProps={{ step: "0.001", min: "0" }}
                value={studyPixelSize}
                onChange={(event) => setStudyPixelSize(event.target.value)}
                error={Boolean(pixelSizeError)}
                helperText={pixelSizeError ?? "Leave blank to clear"}
                sx={{ maxWidth: 220 }}
              />
              <Button variant="outlined" size="small" onClick={handleStudyPixelSizeSave} disabled={pixelSizeMutation.isPending}>
                {pixelSizeMutation.isPending ? "Saving..." : "Save"}
              </Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Controls scale bars for preview panels and downloads.
            </Typography>
          </Stack>
        )}
        {study && (
          <Stack spacing={1}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle1">Study Channels</Typography>
              <Button variant="text" size="small" onClick={resetStudyChannels} disabled={updateChannelsMutation.isPending}>
                Reset to current
              </Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Update channel labels and pseudocolors for this loaded study.
            </Typography>
            {studyChannels.map((channel) => (
              <Box
                key={`study-channel-${channel.channel}`}
                sx={{
                  border: "1px solid rgba(15,23,42,0.08)",
                  borderRadius: 2,
                  p: 1.25,
                  display: "flex",
                  flexDirection: "column",
                  gap: 1
                }}
              >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                  <Typography variant="caption" sx={{ minWidth: 90, fontWeight: 600 }}>
                    Ch{channel.channel}
                  </Typography>
                  <TextField
                    label="Label"
                    size="small"
                    fullWidth
                    value={channel.label}
                    onChange={(event) => handleStudyChannelChange(channel.channel, { label: event.target.value })}
                  />
                  <TextField
                    type="color"
                    label="Color"
                    size="small"
                    value={channel.color}
                    onChange={(event) => handleStudyChannelChange(channel.channel, { color: event.target.value })}
                    sx={{ width: 110 }}
                    inputProps={{ style: { padding: 0, height: 34 } }}
                  />
                </Stack>
              </Box>
            ))}
            <Stack direction="row" spacing={1}>
              <Button
                variant="contained"
                size="small"
                disabled={updateChannelsMutation.isPending}
                onClick={handleStudyChannelSave}
              >
                {updateChannelsMutation.isPending ? "Saving..." : "Save channels"}
              </Button>
            </Stack>
            {updateChannelsMutation.isError && (
              <Alert severity="error">{getErrorMessage(updateChannelsMutation.error)}</Alert>
            )}
            {updateChannelsMutation.isSuccess && (
              <Alert severity="success">Channels updated for this study.</Alert>
            )}
          </Stack>
        )}
        {study && (
          <Stack spacing={1}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle1">Study Ratios</Typography>
              <Button variant="text" size="small" onClick={resetStudyRatios} disabled={updateRatiosMutation.isPending}>
                Reset to current
              </Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Update the ratios available in charts and previews. Changes are saved per study.
            </Typography>
            {studyRatioDrafts.map((ratio, index) => (
              <Box
                key={ratio.id ?? `study-ratio-${index}`}
                sx={{
                  border: "1px solid rgba(15,23,42,0.08)",
                  borderRadius: 2,
                  p: 1.25,
                  display: "flex",
                  flexDirection: "column",
                  gap: 1
                }}
              >
                <TextField
                  label="Label"
                  size="small"
                  fullWidth
                  value={ratio.label}
                  onChange={(event) => handleStudyRatioChange(index, { label: event.target.value })}
                />
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} useFlexGap alignItems={{ sm: "center" }}>
                  <TextField
                    label="Numerator"
                    size="small"
                    select
                    fullWidth
                    sx={{ flex: 1, minWidth: 0 }}
                    value={ratio.numerator_channel}
                    onChange={(event) => handleStudyRatioChange(index, { numerator_channel: Number(event.target.value) })}
                  >
                  {studyChannels.map((channelDefinition) => (
                    <MenuItem key={`study-ratio-num-${index}-${channelDefinition.channel}`} value={channelDefinition.channel}>
                      {studyChannelLabelMap[channelDefinition.channel] ?? `Channel ${channelDefinition.channel}`}
                    </MenuItem>
                  ))}
                  </TextField>
                  <TextField
                    label="Denominator"
                    size="small"
                    select
                    fullWidth
                    sx={{ flex: 1, minWidth: 0 }}
                    value={ratio.denominator_channel}
                    onChange={(event) =>
                      handleStudyRatioChange(index, { denominator_channel: Number(event.target.value) })
                    }
                  >
                  {studyChannels.map((channelDefinition) => (
                    <MenuItem key={`study-ratio-den-${index}-${channelDefinition.channel}`} value={channelDefinition.channel}>
                      {studyChannelLabelMap[channelDefinition.channel] ?? `Channel ${channelDefinition.channel}`}
                    </MenuItem>
                  ))}
                  </TextField>
                  <Button
                    variant="text"
                    size="small"
                    color="error"
                    onClick={() => removeStudyRatio(index)}
                    disabled={studyRatioDrafts.length <= 1}
                    sx={{
                      alignSelf: { xs: "flex-start", sm: "stretch" },
                      whiteSpace: "nowrap"
                    }}
                  >
                    Remove
                  </Button>
                </Stack>
              </Box>
            ))}
            <Stack direction="row" spacing={1}>
              <Button
                variant="outlined"
                size="small"
                onClick={addStudyRatio}
                disabled={studyRatioDrafts.length >= 6 || updateRatiosMutation.isPending}
              >
                Add ratio
              </Button>
              <Button
                variant="contained"
                size="small"
                disabled={updateRatiosMutation.isPending}
                onClick={handleStudyRatioSave}
              >
                {updateRatiosMutation.isPending ? "Saving..." : "Save ratios"}
              </Button>
            </Stack>
            {updateRatiosMutation.isError && (
              <Alert severity="error">{getErrorMessage(updateRatiosMutation.error)}</Alert>
            )}
            {updateRatiosMutation.isSuccess && (
              <Alert severity="success">Ratios updated for this study.</Alert>
            )}
          </Stack>
        )}
      </Stack>
        )}
    </Stack>
  );
}
