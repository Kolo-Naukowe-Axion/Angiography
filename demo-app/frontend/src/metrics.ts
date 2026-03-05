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

export type MaskMetrics = {
  iou: MetricValue;
  dice: MetricValue;
};

type BoxLike = Pick<Box, "x1" | "y1" | "x2" | "y2">;
type PredictionLike = BoxLike & Pick<Box, "confidence">;

type BinaryMask = {
  width: number;
  height: number;
  data: Uint8Array;
};

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

function computeMaskMetrics(prediction: BinaryMask, groundTruth: BinaryMask): MaskMetrics {
  if (prediction.width !== groundTruth.width || prediction.height !== groundTruth.height) {
    return {
      iou: { status: "na", reason: "Prediction/ground-truth mask size mismatch" },
      dice: { status: "na", reason: "Prediction/ground-truth mask size mismatch" },
    };
  }

  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (let index = 0; index < prediction.data.length; index += 1) {
    const pred = prediction.data[index] === 1;
    const gt = groundTruth.data[index] === 1;

    if (pred && gt) {
      tp += 1;
    } else if (pred && !gt) {
      fp += 1;
    } else if (!pred && gt) {
      fn += 1;
    }
  }

  const union = tp + fp + fn;
  const diceDenominator = 2 * tp + fp + fn;

  if (union === 0) {
    return {
      iou: { status: "ready", value: 1 },
      dice: { status: "ready", value: 1 },
    };
  }

  return {
    iou: { status: "ready", value: tp / union },
    dice: { status: "ready", value: diceDenominator === 0 ? 1 : (2 * tp) / diceDenominator },
  };
}

async function loadBinaryMask(url: string): Promise<BinaryMask> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Mask request failed: ${response.status} ${response.statusText}`);
  }

  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);

  const canvas = document.createElement("canvas");
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;

  const context = canvas.getContext("2d");
  if (!context) {
    imageBitmap.close();
    throw new Error("Could not create canvas context for mask decoding.");
  }

  context.drawImage(imageBitmap, 0, 0);
  imageBitmap.close();

  const imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const binary = new Uint8Array(canvas.width * canvas.height);

  for (let pixel = 0; pixel < binary.length; pixel += 1) {
    const alpha = imageData[pixel * 4 + 3];
    const value = imageData[pixel * 4];
    binary[pixel] = alpha > 0 && value >= 128 ? 1 : 0;
  }

  return {
    width: canvas.width,
    height: canvas.height,
    data: binary,
  };
}

export async function computeMaskMetricsFromUrls(predictionUrl: string, groundTruthUrl: string): Promise<MaskMetrics> {
  const [predictionMask, groundTruthMask] = await Promise.all([
    loadBinaryMask(predictionUrl),
    loadBinaryMask(groundTruthUrl),
  ]);

  return computeMaskMetrics(predictionMask, groundTruthMask);
}
