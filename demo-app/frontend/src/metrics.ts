import type { Box } from "./types";

const DEFAULT_IOU_THRESHOLD = 0.5;
const NO_LABELS_REASON = "No labels available";

export type MetricStatus = "ready" | "na";

export type MetricValue =
  | {
      status: "ready";
      value: number;
    }
  | {
      status: "na";
      reason: string;
    };

export type MatchPair = {
  predictionIndex: number;
  groundTruthIndex: number;
  confidence: number;
  iou: number;
};

export type PredictionMatch = {
  predictionIndex: number;
  confidence: number;
  isTruePositive: boolean;
  matchedGroundTruthIndex: number | null;
  iou: number;
};

export type FrameMatchResult = {
  predictions: PredictionMatch[];
  matches: MatchPair[];
  unmatchedGroundTruthIndices: number[];
};

export type FrameMetrics = {
  iou: MetricValue;
  matchCount: number;
  predictionCount: number;
  groundTruthCount: number;
};

type BoxLike = Pick<Box, "x1" | "y1" | "x2" | "y2">;
type PredictionLike = BoxLike & Pick<Box, "confidence">;

export function computeIoU(a: BoxLike, b: BoxLike): number {
  const intersectionX1 = Math.max(a.x1, b.x1);
  const intersectionY1 = Math.max(a.y1, b.y1);
  const intersectionX2 = Math.min(a.x2, b.x2);
  const intersectionY2 = Math.min(a.y2, b.y2);

  const intersectionWidth = Math.max(0, intersectionX2 - intersectionX1);
  const intersectionHeight = Math.max(0, intersectionY2 - intersectionY1);
  const intersectionArea = intersectionWidth * intersectionHeight;

  const aArea = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const bArea = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const unionArea = aArea + bArea - intersectionArea;

  if (unionArea <= 0) {
    return 0;
  }

  return intersectionArea / unionArea;
}

export function greedyMatchPredictions(
  predictions: readonly PredictionLike[],
  groundTruth: readonly BoxLike[],
  iouThreshold: number = DEFAULT_IOU_THRESHOLD,
): FrameMatchResult {
  const sortedPredictionIndices = predictions
    .map((_, index) => index)
    .sort((left, right) => predictions[right].confidence - predictions[left].confidence || left - right);

  const matchedGroundTruth = new Set<number>();
  const predictionMatches: PredictionMatch[] = [];
  const matches: MatchPair[] = [];

  for (const predictionIndex of sortedPredictionIndices) {
    const prediction = predictions[predictionIndex];

    let bestGroundTruthIndex = -1;
    let bestIou = 0;

    for (let gtIndex = 0; gtIndex < groundTruth.length; gtIndex += 1) {
      if (matchedGroundTruth.has(gtIndex)) {
        continue;
      }

      const iou = computeIoU(prediction, groundTruth[gtIndex]);
      if (iou > bestIou) {
        bestIou = iou;
        bestGroundTruthIndex = gtIndex;
      }
    }

    if (bestGroundTruthIndex >= 0 && bestIou >= iouThreshold) {
      matchedGroundTruth.add(bestGroundTruthIndex);
      matches.push({
        predictionIndex,
        groundTruthIndex: bestGroundTruthIndex,
        confidence: prediction.confidence,
        iou: bestIou,
      });
      predictionMatches.push({
        predictionIndex,
        confidence: prediction.confidence,
        isTruePositive: true,
        matchedGroundTruthIndex: bestGroundTruthIndex,
        iou: bestIou,
      });
      continue;
    }

    predictionMatches.push({
      predictionIndex,
      confidence: prediction.confidence,
      isTruePositive: false,
      matchedGroundTruthIndex: null,
      iou: bestIou,
    });
  }

  const unmatchedGroundTruthIndices = groundTruth
    .map((_, index) => index)
    .filter((index) => !matchedGroundTruth.has(index));

  return {
    predictions: predictionMatches,
    matches,
    unmatchedGroundTruthIndices,
  };
}

export function computeBestIoU(predictions: readonly BoxLike[], groundTruth: readonly BoxLike[]): number {
  if (groundTruth.length === 0) {
    return predictions.length === 0 ? 1 : 0;
  }

  if (predictions.length === 0) {
    return 0;
  }

  let bestIou = 0;
  for (const prediction of predictions) {
    for (const gt of groundTruth) {
      bestIou = Math.max(bestIou, computeIoU(prediction, gt));
    }
  }

  return bestIou;
}

export function computeMeanIoU(
  matches: readonly MatchPair[],
  predictionCount: number,
  groundTruthCount: number,
): number {
  if (groundTruthCount === 0) {
    return predictionCount === 0 ? 1 : 0;
  }

  if (matches.length === 0) {
    return 0;
  }

  const totalIou = matches.reduce((sum, match) => sum + match.iou, 0);
  return totalIou / matches.length;
}

export function computeFrameMetrics({
  predictions,
  groundTruth,
  labelsAvailable,
  iouThreshold = DEFAULT_IOU_THRESHOLD,
}: {
  predictions: readonly PredictionLike[];
  groundTruth: readonly BoxLike[];
  labelsAvailable: boolean;
  iouThreshold?: number;
}): FrameMetrics {
  if (!labelsAvailable) {
    return {
      iou: { status: "na", reason: NO_LABELS_REASON },
      matchCount: 0,
      predictionCount: predictions.length,
      groundTruthCount: groundTruth.length,
    };
  }

  const iou = computeBestIoU(predictions, groundTruth);
  const matchResult = greedyMatchPredictions(predictions, groundTruth, iouThreshold);

  return {
    iou: { status: "ready", value: iou },
    matchCount: matchResult.matches.length,
    predictionCount: predictions.length,
    groundTruthCount: groundTruth.length,
  };
}
