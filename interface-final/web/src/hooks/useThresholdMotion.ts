import { useEffect, useRef, useState } from "react";
import type { AnalyzeResponse, IndividualImageRecord, MouseAverageRecord } from "../api/types";

const THRESHOLD_MOTION_DURATION_MS = 2000;
const REPLICA_REVEAL_END = 0.18;
const REPLICA_MOVE_START = 0.12;
const REPLICA_MOVE_END = 0.68;
const SUBJECT_MOVE_START = 0.28;
const SUBJECT_MOVE_END = 0.78;
const REPLICA_HIDE_START = 0.82;

export type ThresholdMotionStage = {
  replicaVisibility: number;
  replicaProgress: number;
  subjectProgress: number;
};

export type ThresholdMotionState = {
  active: boolean;
  progress: number;
  stage: ThresholdMotionStage;
  displayData?: AnalyzeResponse;
  fromData?: AnalyzeResponse;
  toData?: AnalyzeResponse;
};

const idleStage: ThresholdMotionStage = {
  replicaVisibility: 0,
  replicaProgress: 1,
  subjectProgress: 1
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const easeOutCubic = (value: number) => 1 - (1 - value) ** 3;

const easeInOutCubic = (value: number) =>
  value < 0.5 ? 4 * value ** 3 : 1 - ((-2 * value + 2) ** 3) / 2;

const interpolateNumber = (from: number, to: number, progress: number) => from + (to - from) * progress;

export const thresholdsEqual = (left: Record<string, number>, right: Record<string, number>) => {
  const keys = Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).sort();
  return keys.every((key) => left[key] === right[key]);
};

const buildRatioKeys = (fromRatios?: Record<string, number>, toRatios?: Record<string, number>) =>
  Array.from(new Set([...Object.keys(fromRatios ?? {}), ...Object.keys(toRatios ?? {})]));

const interpolateRatios = (
  fromRatios: Record<string, number> | undefined,
  toRatios: Record<string, number> | undefined,
  progress: number
) =>
  buildRatioKeys(fromRatios, toRatios).reduce<Record<string, number>>((acc, key) => {
    const target = toRatios?.[key];
    const start = fromRatios?.[key];
    if (typeof target !== "number" || !Number.isFinite(target)) {
      if (typeof start === "number" && Number.isFinite(start)) {
        acc[key] = start;
      }
      return acc;
    }
    if (typeof start !== "number" || !Number.isFinite(start)) {
      acc[key] = target;
      return acc;
    }
    acc[key] = interpolateNumber(start, target, progress);
    return acc;
  }, {});

const mouseAverageKey = (record: MouseAverageRecord) => `${record.Group}|${record.MouseID}`;

const individualImageKey = (record: IndividualImageRecord) =>
  `${record.group}|${record.mouse_id}|${record.filename}|${record.replicate_index}`;

export const getThresholdMotionStage = (progress: number): ThresholdMotionStage => {
  const normalized = clamp01(progress);
  let replicaVisibility: number;

  if (normalized <= REPLICA_REVEAL_END) {
    replicaVisibility = easeOutCubic(clamp01(normalized / REPLICA_REVEAL_END));
  } else if (normalized >= REPLICA_HIDE_START) {
    replicaVisibility = 1 - easeInOutCubic(clamp01((normalized - REPLICA_HIDE_START) / (1 - REPLICA_HIDE_START)));
  } else {
    replicaVisibility = 1;
  }

  return {
    replicaVisibility,
    replicaProgress: easeInOutCubic(
      clamp01((normalized - REPLICA_MOVE_START) / (REPLICA_MOVE_END - REPLICA_MOVE_START))
    ),
    subjectProgress: easeInOutCubic(
      clamp01((normalized - SUBJECT_MOVE_START) / (SUBJECT_MOVE_END - SUBJECT_MOVE_START))
    )
  };
};

const buildChannelAreaKeys = (
  fromAreas?: Record<string, number>,
  toAreas?: Record<string, number>
) => Array.from(new Set([...Object.keys(fromAreas ?? {}), ...Object.keys(toAreas ?? {})])).sort();

const interpolateChannelAreas = (
  fromAreas: Record<string, number> | undefined,
  toAreas: Record<string, number> | undefined,
  progress: number
) =>
  buildChannelAreaKeys(fromAreas, toAreas).reduce<Record<string, number>>((acc, key) => {
    const target = toAreas?.[key];
    const start = fromAreas?.[key];
    if (typeof target !== "number" || !Number.isFinite(target)) {
      if (typeof start === "number" && Number.isFinite(start)) {
        acc[key] = start;
      }
      return acc;
    }
    if (typeof start !== "number" || !Number.isFinite(start)) {
      acc[key] = target;
      return acc;
    }
    acc[key] = interpolateNumber(start, target, progress);
    return acc;
  }, {});

const interpolateMouseAverage = (
  fromRecord: MouseAverageRecord | undefined,
  toRecord: MouseAverageRecord,
  progress: number
): MouseAverageRecord => {
  if (!fromRecord) {
    return toRecord;
  }

  const nextRecord: MouseAverageRecord = {
    Group: toRecord.Group,
    MouseID: toRecord.MouseID,
    channel_areas: interpolateChannelAreas(fromRecord.channel_areas, toRecord.channel_areas, progress)
  };

  const ratios = interpolateRatios(fromRecord.ratios, toRecord.ratios, progress);
  if (Object.keys(ratios).length > 0) {
    nextRecord.ratios = ratios;
  }

  return nextRecord;
};

const interpolateIndividualImage = (
  fromRecord: IndividualImageRecord | undefined,
  toRecord: IndividualImageRecord,
  progress: number
): IndividualImageRecord => {
  if (!fromRecord) {
    return toRecord;
  }

  const nextRecord: IndividualImageRecord = {
    ...toRecord,
    channel_areas: interpolateChannelAreas(fromRecord.channel_areas, toRecord.channel_areas, progress),
    ratios: {}
  };

  nextRecord.ratios = interpolateRatios(fromRecord.ratios, toRecord.ratios, progress);

  return nextRecord;
};

const interpolateAnalysis = (
  fromData: AnalyzeResponse,
  toData: AnalyzeResponse,
  stage: ThresholdMotionStage
): AnalyzeResponse => {
  const previousMouseAverages = new Map(fromData.mouse_averages.map((record) => [mouseAverageKey(record), record]));
  const previousIndividualImages = new Map(
    fromData.individual_images.map((record) => [individualImageKey(record), record])
  );

  return {
    ...toData,
    mouse_averages: toData.mouse_averages.map((record) =>
      interpolateMouseAverage(previousMouseAverages.get(mouseAverageKey(record)), record, stage.subjectProgress)
    ),
    individual_images: toData.individual_images.map((record) =>
      interpolateIndividualImage(previousIndividualImages.get(individualImageKey(record)), record, stage.replicaProgress)
    )
  };
};

export function useThresholdMotion(data: AnalyzeResponse | undefined): ThresholdMotionState {
  const [state, setState] = useState<ThresholdMotionState>({
    active: false,
    progress: 1,
    stage: idleStage,
    displayData: data,
    fromData: data,
    toData: data
  });
  const frameRef = useRef<number | null>(null);
  const presentedDataRef = useRef<AnalyzeResponse | undefined>(data);

  useEffect(() => {
    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!data) {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      presentedDataRef.current = undefined;
      setState({
        active: false,
        progress: 1,
        stage: idleStage,
        displayData: undefined,
        fromData: undefined,
        toData: undefined
      });
      return;
    }

    if (
      !presentedDataRef.current ||
      presentedDataRef.current.study_id !== data.study_id ||
      thresholdsEqual(presentedDataRef.current.thresholds, data.thresholds)
    ) {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      presentedDataRef.current = data;
      setState({
        active: false,
        progress: 1,
        stage: idleStage,
        displayData: data,
        fromData: data,
        toData: data
      });
      return;
    }

    const fromData = presentedDataRef.current;
    const toData = data;
    const startTime = performance.now();

    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
    }

    const step = (timestamp: number) => {
      const progress = clamp01((timestamp - startTime) / THRESHOLD_MOTION_DURATION_MS);
      const stage = getThresholdMotionStage(progress);
      const displayData = interpolateAnalysis(fromData, toData, stage);
      presentedDataRef.current = displayData;

      if (progress < 1) {
        setState({
          active: true,
          progress,
          stage,
          displayData,
          fromData,
          toData
        });
        frameRef.current = window.requestAnimationFrame(step);
        return;
      }

      presentedDataRef.current = toData;
      frameRef.current = null;
      setState({
        active: false,
        progress: 1,
        stage: idleStage,
        displayData: toData,
        fromData: toData,
        toData
      });
    };

    frameRef.current = window.requestAnimationFrame(step);
  }, [data]);

  return state;
}
