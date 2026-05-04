import { NextRequest, NextResponse } from 'next/server';
import modelData from './model.json';

// ── TypeScript XGBoost evaluator ──────────────────────────────────────────────
// Mirrors the binary:logistic XGBoost JSON format produced by model.save_model()

interface XGBTree {
  left_children:   number[];
  right_children:  number[];
  split_indices:   number[];
  split_conditions: number[];
  base_weights:    number[];
  default_left:    number[];
}

function traverseTree(tree: XGBTree, features: number[]): number {
  let node = 0;
  while (tree.left_children[node] !== -1) {
    const featureVal = features[tree.split_indices[node]];
    const goLeft = featureVal < tree.split_conditions[node]; // XGBoost uses strict <
    node = goLeft ? tree.left_children[node] : tree.right_children[node];
  }
  return tree.base_weights[node];
}

function predictProba(features: number[]): number {
  const trees = (modelData as any).learner.gradient_booster.model.trees as XGBTree[];
  // base_score=0.5 → logit(0.5)=0, so initial margin is 0
  let margin = 0;
  for (const tree of trees) {
    margin += traverseTree(tree, features);
  }
  // sigmoid
  return 1 / (1 + Math.exp(-margin));
}

// Feature order must match training: complexity_score, estimated_steps,
// priority_encoded, avg_gaze, avg_head_pose, avg_eye_openness, gaze_variance
const FEATURE_ORDER = [
  'complexity_score',
  'estimated_steps',
  'priority_encoded',
  'avg_gaze',
  'avg_head_pose',
  'avg_eye_openness',
  'gaze_variance',
] as const;

export async function POST(req: NextRequest) {
  const body = await req.json();
  const features = FEATURE_ORDER.map(k => Number(body[k] ?? 0));
  const riskScore = predictProba(features);
  return NextResponse.json({ riskScore });
}
