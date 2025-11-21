import { useEffect, useMemo, useRef, useState } from "react";
import { isAxiosError } from "axios";
import {
  Alert,
  Box,
  Button,
  Chip,
  Checkbox,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Stack,
  Switch,
  TextField,
  Typography
} from "@mui/material";
import { useAppStore } from "../state/useAppStore";
import {
  useConfigCreate,
  useConfigRead,
  useConfigScan,
  useFileUpload,
  useLoadStudy,
  useRunStatus,
  useThresholdRun,
  useUpdateRatios,
  usePixelSizeUpdate
} from "../api/hooks";
import type { ConfigScanResponse, RatioDefinition } from "../api/types";
import { DEFAULT_RATIO_DEFINITIONS } from "../constants/metrics";

const sliderMarks = [0, 1000, 2000, 3000, 4000].map((value) => ({
  value,
  label: String(value)
}));

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

type StepId = "nd2" | "config" | "run" | "load";
type GuideState = "completed" | "active" | "upcoming";

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

export default function LeftPanel() {
  const {
    study,
    thresholds,
    statisticsEnabled,
    statisticsSettings,
    plotSettings,
    ratioDefinitions,
    setThreshold,
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
    previewChannelRanges,
    setPreviewChannelRange,
    resetPreviewChannelRanges,
    updateStudy
  } = useAppStore();
  const [inputDir, setInputDir] = useState("");
  const [configPath, setConfigPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [resultsPath, setResultsPath] = useState("");
  const [configOriginalName, setConfigOriginalName] = useState<string | null>(null);
  const [resultsOriginalName, setResultsOriginalName] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<ConfigScanResponse | null>(null);
  const [groupsJson, setGroupsJson] = useState("{}");
  const [pixelSize, setPixelSize] = useState("");
  const [configError, setConfigError] = useState<string | null>(null);
  const [groupMap, setGroupMap] = useState<Record<string, string[]>>({});
  const [newGroupName, setNewGroupName] = useState("");
  const [subjectInputs, setSubjectInputs] = useState<Record<string, string>>({});
  const [pairSelection, setPairSelection] = useState<{ first: string; second: string }>({ first: "", second: "" });
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [ratioDrafts, setRatioDrafts] = useState<RatioDefinition[]>(DEFAULT_RATIO_DEFINITIONS);
  const [studyRatioDrafts, setStudyRatioDrafts] = useState<RatioDefinition[]>(DEFAULT_RATIO_DEFINITIONS);
  const [palettePresetOverride, setPalettePresetOverride] = useState<string | null>(null);
  const [studyPixelSize, setStudyPixelSize] = useState("");
  const [pixelSizeError, setPixelSizeError] = useState<string | null>(null);

  const scanMutation = useConfigScan();
  const configMutation = useConfigCreate();
  const configReadMutation = useConfigRead();
  const runMutation = useThresholdRun();
  const loadMutation = useLoadStudy();
  const statusQuery = useRunStatus(jobId);
  const uploadMutation = useFileUpload();
  const updateRatiosMutation = useUpdateRatios(study?.study_id ?? null);
  const pixelSizeMutation = usePixelSizeUpdate(study?.study_id ?? null);
  const configInputRef = useRef<HTMLInputElement>(null);
  const resultsInputRef = useRef<HTMLInputElement>(null);

  const studyGroups = useMemo(() => study?.groups ?? [], [study?.groups]);

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

  useEffect(() => {
    setPairSelection({ first: "", second: "" });
  }, [statisticsSettings.comparisonMode, study?.study_id]);

  const scanDefaults = useMemo(() => {
    if (!scanResult) return {};
    const defaults: Record<string, string[]> = {};
    scanResult.groups.forEach((group) => {
      defaults[group.group_name] = group.subjects.map((subject) => subject.subject_id);
    });
    return defaults;
  }, [scanResult]);

  const statsControlsDisabled = !statisticsEnabled || studyGroups.length === 0;
  const canAddPair =
    statisticsEnabled &&
    statisticsSettings.comparisonMode === "pairs" &&
    pairSelection.first &&
    pairSelection.second &&
    pairSelection.first !== pairSelection.second;

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

  const handleAddPair = () => {
    if (!canAddPair) {
      return;
    }
    addComparisonPair([pairSelection.first, pairSelection.second]);
    setPairSelection({ first: "", second: "" });
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
    return Math.max(1, Math.min(3, Math.round(numeric)));
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
        denominator_channel: 3
      }
    ]);
  };

  const handleConfigRatioChange = (index: number, patch: Partial<RatioDefinition>) =>
    updateRatioDraft(setRatioDrafts, index, patch);
  const handleStudyRatioChange = (index: number, patch: Partial<RatioDefinition>) =>
    updateRatioDraft(setStudyRatioDrafts, index, patch);
  const addConfigRatio = () => addRatioDraft(setRatioDrafts);
  const addStudyRatio = () => addRatioDraft(setStudyRatioDrafts);
  const removeConfigRatio = (index: number) => removeRatioDraft(setRatioDrafts, index);
  const removeStudyRatio = (index: number) => removeRatioDraft(setStudyRatioDrafts, index);
  const resetConfigRatios = () => setRatioDrafts(DEFAULT_RATIO_DEFINITIONS);
  const resetStudyRatios = () => setStudyRatioDrafts(ratioDefinitions);
  const handleStudyRatioSave = async () => {
    if (!study) return;
    try {
      const response = await updateRatiosMutation.mutateAsync(studyRatioDrafts);
      setRatioDefinitions(response.ratios);
    } catch (error) {
      /* handled below */
    }
  };

  const clampIntensity = (value: number) => Math.max(0, Math.min(4095, Math.round(value)));
  const applyChannelWindow = (
    channel: "channel_1" | "channel_2" | "channel_3",
    next: { min?: number; threshold?: number; max?: number }
  ) => {
    const [currentMin, currentMax] = previewChannelRanges[channel];
    const min = clampIntensity(next.min ?? currentMin);
    const maxCandidate = clampIntensity(next.max ?? currentMax);
    const resolvedMin = Math.min(min, maxCandidate);
    const resolvedMax = Math.max(resolvedMin + 1, maxCandidate);
    const thresholdValue = clampIntensity(next.threshold ?? thresholds[channel]);
    const clampedThreshold = Math.min(Math.max(thresholdValue, resolvedMin), resolvedMax);
    setPreviewChannelRange(channel, [resolvedMin, resolvedMax]);
    setThreshold(channel, clampedThreshold);
  };

  const removeRatioDraft = (setter: React.Dispatch<React.SetStateAction<RatioDefinition[]>>, index: number) => {
    setter((prev) => prev.filter((_, idx) => idx !== index));
  };

  const availableGroupNames = useMemo(() => Object.keys(groupMap).sort(), [groupMap]);
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
    const nd2Complete = Boolean(inputDir.trim());
    const configComplete = Boolean(configPath && hasGroups);
    const runComplete =
      Boolean(resultsPath) || Boolean(statusQuery.data?.state === "succeeded" && statusQuery.data.output_path);
    const loadComplete = Boolean(study);
    const base: Array<{ id: StepId; title: string; description: string; completed: boolean }> = [
      {
        id: "nd2",
        title: "Input ND2 directory",
        description: "Point to the folder with ND2 files, then scan for subjects.",
        completed: nd2Complete
      },
      {
        id: "config",
        title: "Configure groups",
        description: "Edit detected cohorts and save a config JSON.",
        completed: configComplete
      },
      {
        id: "run",
        title: "Run thresholds",
        description: "Generate threshold results for the current config.",
        completed: runComplete
      },
      {
        id: "load",
        title: "Load study",
        description: "Load the latest results JSON to unlock analysis.",
        completed: loadComplete
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
    inputDir,
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

  return (
    <Stack spacing={2} px={3} py={3} sx={{ minHeight: "100%" }}>
      {!study && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Study Pipeline
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Provide the ND2 directory and optional config path. The interface can scan the folder, create a configuration, run threshold generation, and load the results—all from here.
          </Typography>
        </Box>
      )}

      {!study && (
        <Box
          sx={{
            borderRadius: 2,
            border: "1px solid rgba(15,23,42,0.1)",
            p: 2,
            backgroundColor: "#ffffff"
          }}
        >
          <Typography variant="subtitle2" gutterBottom>
            Interactive workflow
          </Typography>
          <Stack spacing={1}>
            {workflowSteps.map((step, index) => (
              <Box
                key={step.id}
                sx={{ borderRadius: 1.5, p: 1, backgroundColor: step.state === "active" ? "rgba(37,99,235,0.08)" : "transparent" }}
              >
                <Stack direction="row" spacing={1} alignItems="center">
                  <Chip
                    label={`Step ${index + 1}`}
                    size="small"
                    color={step.state === "completed" ? "success" : step.state === "active" ? "primary" : "default"}
                    variant={step.state === "upcoming" ? "outlined" : "filled"}
                  />
                  <Typography variant="body2">{step.title}</Typography>
                </Stack>
                <Typography variant="caption" color="text.secondary" sx={{ ml: 5 }}>
                  {step.description}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Box>
      )}


      <Box sx={sectionSx("nd2")}>
        <Stack spacing={1.5}>
          <TextField
            label="ND2 Input Directory"
            value={inputDir}
            onChange={(event) => setInputDir(event.target.value)}
            size="small"
            fullWidth
            placeholder="/path/to/nd2"
          />
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              size="small"
              disabled={!inputDir || scanMutation.isPending}
              onClick={async () => {
                try {
                  const response = await scanMutation.mutateAsync({ input_dir: inputDir });
                  setScanResult(response);
                  const groups: Record<string, string[]> = {};
                  response.groups.forEach((group) => {
                    groups[group.group_name] = group.subjects.map((subject) => subject.subject_id);
                  });
                  const normalized = normalizeGroupMapping(groups);
                  setGroupMap(normalized);
                  setGroupsJson(JSON.stringify(normalized, null, 2));
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
          <Typography variant="caption" color="text.secondary">
            {builderSummary.groupCount} editable groups • {builderSummary.subjectCount} subjects in the builder.
          </Typography>
          {scanSummary && (
            <Typography variant="caption" color="text.secondary">
              Last scan detected {scanSummary.groupCount} groups across {scanSummary.subjectCount} subjects ({scanSummary.files} ND2 files).
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
      </Box>

      <Box sx={sectionSx("config")}>
        {scanResult || hasGroups ? (
          <Stack spacing={1.75}>
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
                      ratios: ratioDrafts
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
            <Stack spacing={1}>
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
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr", sm: "1fr repeat(2, 160px) auto" },
                    gap: 8,
                    alignItems: "center"
                  }}
                >
                  <TextField
                    label="Label"
                    size="small"
                    fullWidth
                    value={ratio.label}
                    onChange={(event) => handleConfigRatioChange(index, { label: event.target.value })}
                  />
                  <TextField
                    label="Numerator"
                    size="small"
                    type="number"
                    inputProps={{ min: 1, max: 3 }}
                    value={ratio.numerator_channel}
                    onChange={(event) => handleConfigRatioChange(index, { numerator_channel: Number(event.target.value) })}
                  />
                  <TextField
                    label="Denominator"
                    size="small"
                    type="number"
                    inputProps={{ min: 1, max: 3 }}
                    value={ratio.denominator_channel}
                    onChange={(event) =>
                      handleConfigRatioChange(index, { denominator_channel: Number(event.target.value) })
                    }
                  />
                  <Button
                    variant="text"
                    size="small"
                    color="error"
                    onClick={() => removeConfigRatio(index)}
                    disabled={ratioDrafts.length <= 1}
                    sx={{ justifySelf: { xs: "flex-start", sm: "center" } }}
                  >
                    Remove
                  </Button>
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
            Scan an ND2 directory to unlock group editing, JSON overrides, ratio presets, and config exports.
          </Typography>
        )}
      </Box>

      <Divider flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />

      <Box sx={sectionSx("run")}>
        <Stack spacing={1.5}>
        <Typography variant="subtitle1">Threshold &amp; Range</Typography>
        <Typography variant="caption" color="text.secondary">
          Threshold only controls the mask; Min/Max control the raw intensity window for live previews. Overlay combines both.
        </Typography>
        {(["channel_1", "channel_2", "channel_3"] as const).map((channel, index) => {
          const range = previewChannelRanges[channel];
          const thresholdValue = Math.min(Math.max(thresholds[channel], range[0]), range[1]);
          return (
            <Box key={channel}>
              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                <Typography variant="caption" color="text.secondary">
                  {`Channel ${index + 1}`}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {`Min ${range[0]} • Thr ${thresholdValue} • Max ${range[1]}`}
                </Typography>
              </Stack>
              <Slider
                size="small"
                min={0}
                max={4095}
                marks={sliderMarks}
                disableSwap
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
                value={[range[0], thresholdValue, range[1]]}
                onChange={(_, value) => {
                  if (!Array.isArray(value) || value.length !== 3) {
                    return;
                  }
                  applyChannelWindow(channel, { min: value[0], threshold: value[1], max: value[2] });
                }}
              />
              <Stack direction="row" spacing={1}>
                <TextField
                  size="small"
                  type="number"
                  label="Min"
                  value={range[0]}
                  inputProps={{ min: 0, max: 4095 }}
                  onChange={(event) => applyChannelWindow(channel, { min: Number(event.target.value) })}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Threshold"
                  value={thresholdValue}
                  inputProps={{ min: 0, max: 4095 }}
                  onChange={(event) => applyChannelWindow(channel, { threshold: Number(event.target.value) })}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Max"
                  value={range[1]}
                  inputProps={{ min: 1, max: 4095 }}
                  onChange={(event) => applyChannelWindow(channel, { max: Number(event.target.value) })}
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
                <Stack direction="row" spacing={1}>
                  <TextField
                    select
                    size="small"
                    label="Group A"
                    value={pairSelection.first}
                    onChange={(event) =>
                      setPairSelection((prev) => ({ ...prev, first: event.target.value as string }))
                    }
                    disabled={statsControlsDisabled}
                    InputLabelProps={{ shrink: true }}
                    SelectProps={{
                      displayEmpty: true,
                      renderValue: (selected) => (selected ? String(selected) : "Select group")
                    }}
                    fullWidth
                  >
                    <MenuItem value="" disabled>
                      Select group
                    </MenuItem>
                    {studyGroups.map((group) => (
                      <MenuItem key={group} value={group}>
                        {group}
                      </MenuItem>
                    ))}
                  </TextField>
                  <TextField
                    select
                    size="small"
                    label="Group B"
                    value={pairSelection.second}
                    onChange={(event) =>
                      setPairSelection((prev) => ({ ...prev, second: event.target.value as string }))
                    }
                    disabled={statsControlsDisabled}
                    InputLabelProps={{ shrink: true }}
                    SelectProps={{
                      displayEmpty: true,
                      renderValue: (selected) => (selected ? String(selected) : "Select group")
                    }}
                    fullWidth
                  >
                    <MenuItem value="" disabled>
                      Select group
                    </MenuItem>
                    {studyGroups.map((group) => (
                      <MenuItem key={group} value={group}>
                        {group}
                      </MenuItem>
                    ))}
                  </TextField>
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Button variant="outlined" size="small" disabled={!canAddPair} onClick={handleAddPair}>
                    Add pair
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
                  {statisticsSettings.comparisonPairs.map((pair) => {
                    const key = `${pair[0]}-${pair[1]}`;
                    return (
                      <Chip
                        key={key}
                        size="small"
                        label={`${pair[0]} ↔ ${pair[1]}`}
                        onDelete={
                          statisticsEnabled ? () => removeComparisonPair(pair) : undefined
                        }
                      />
                    );
                  })}
                  {statisticsSettings.comparisonPairs.length === 0 && (
                    <Typography variant="caption" color="text.secondary">
                      Add pairs to compare specific groups.
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
      </Box>

      <Divider flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />

      <Box sx={sectionSx("load")}>
        <Stack spacing={1.5}>
        <Typography variant="subtitle1">Visualization Settings</Typography>
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
      </Box>

      <Divider flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />

      <Box sx={sectionSx("run")}>
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
          {runMutation.isPending ? "Launching..." : "Run Threshold Generation"}
        </Button>
        {statusQuery.data && (
          <Alert severity={statusQuery.data.state === "failed" ? "error" : statusQuery.data.state === "succeeded" ? "success" : "info"}>
            {statusQuery.data.state.toUpperCase()}: {statusQuery.data.message ?? "Processing"}
          </Alert>
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
      </Box>

      <Divider flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />

      <Box sx={sectionSx("load")}>
      <Box sx={sectionSx("load")}>
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
            ND2 directory unavailable at {study.nd2_root}. Mount or copy the folder, update “ND2 Input Directory”, and load the study
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
                    type="number"
                    fullWidth
                    sx={{ flex: 1, minWidth: 0 }}
                    inputProps={{ min: 1, max: 3 }}
                    value={ratio.numerator_channel}
                    onChange={(event) => handleStudyRatioChange(index, { numerator_channel: Number(event.target.value) })}
                  />
                  <TextField
                    label="Denominator"
                    size="small"
                    type="number"
                    fullWidth
                    sx={{ flex: 1, minWidth: 0 }}
                    inputProps={{ min: 1, max: 3 }}
                    value={ratio.denominator_channel}
                    onChange={(event) =>
                      handleStudyRatioChange(index, { denominator_channel: Number(event.target.value) })
                    }
                  />
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
      </Box>


      </Box>

      <Divider flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />
    </Stack>
  );
}
