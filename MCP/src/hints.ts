import type { Area, Problem } from "./types.js";

const AREA_GUIDANCE: Record<Area, [string, string, string]> = {
  "general-coding": [
    "Restate the input, output, constraints, and two edge cases before choosing a data structure.",
    "Name the dominant operation and compare a direct approach with one standard pattern that improves it.",
    "Walk through one small example, then state time and space complexity and test boundary conditions.",
  ],
  "ml-coding": [
    "Write down tensor or array shapes, expected output, and the numerical edge cases first.",
    "Separate the mathematical definition from vectorization, batching, and implementation details.",
    "Check invariants, numerical stability, empty or degenerate inputs, and the cost of each operation.",
  ],
  pytorch: [
    "Annotate tensor shapes, dtype, device, gradient requirements, and training versus evaluation behavior.",
    "Decompose the task into module state, forward computation, loss, backward pass, and optimizer update.",
    "Test shape and device consistency, gradient flow, parameter registration, and checkpoint behavior.",
  ],
  "ml-fundamentals": [
    "Start with the objective, assumptions, inputs, outputs, and the failure mode the concept addresses.",
    "Explain the mechanism, then contrast it with the nearest alternative using one concrete example.",
    "Discuss evaluation, data limitations, computational tradeoffs, and what would invalidate your assumptions.",
  ],
  genai: [
    "Clarify the modality, task, context window, supervision source, and evaluation target.",
    "Trace the data and representation flow through the model, training objective, and inference procedure.",
    "Compare quality, latency, cost, safety, and controllability; include one likely failure and mitigation.",
  ],
  "system-design": [
    "Clarify users, business objective, scale, latency, freshness, safety, and the online decision being made.",
    "Sketch data collection, labels, features, model, serving path, feedback loop, and offline and online evaluation.",
    "Pressure-test cold start, drift, abuse, reliability, experimentation, and cost; prioritize the largest risk.",
  ],
  behavioral: [
    "Choose one specific situation and state the stakes, your ownership, and the people involved.",
    "Use Situation, Task, Action, Result; keep the Action section focused on decisions you personally made.",
    "Quantify the result, explain the tradeoff, and close with what you learned or would change.",
  ],
};

export function getProgressiveHint(problem: Problem, level: 1 | 2 | 3): string {
  const guidance = AREA_GUIDANCE[problem.area][level - 1];
  const prefix = [
    "Clarify before solving.",
    "Choose and justify an approach.",
    "Validate without revealing the solution.",
  ][level - 1];
  return `${prefix} ${guidance}`;
}

export function getAnswerReviewChecklist(problem: Problem): string[] {
  const common = [
    "Did the answer address the exact question and state its assumptions?",
    "Did it explain why the chosen approach is appropriate?",
    "Did it identify important edge cases or failure modes?",
  ];
  const byArea: Record<Area, string[]> = {
    "general-coding": [
      "Does the reasoning include correctness plus time and space complexity?",
      "Are boundary cases covered with concrete tests?",
    ],
    "ml-coding": [
      "Are shapes, vectorization, numerical stability, and complexity covered?",
      "Are tests proposed for typical and degenerate inputs?",
    ],
    pytorch: [
      "Are shape, dtype, device, gradient flow, and train/eval behavior covered?",
      "Would parameters and state be registered and checkpointed correctly?",
    ],
    "ml-fundamentals": [
      "Are the objective, assumptions, mechanism, and comparison point correct and connected?",
      "Does the answer discuss data and evaluation limitations?",
    ],
    genai: [
      "Are training and inference explained separately?",
      "Are modality-specific evaluation, safety, latency, and cost tradeoffs covered?",
    ],
    "system-design": [
      "Does the design cover data, labels, modeling, serving, evaluation, and feedback loops?",
      "Are scale, reliability, drift, safety, and experimentation addressed?",
    ],
    behavioral: [
      "Is personal ownership clear rather than hidden behind 'we'?",
      "Is the result measurable, and is the reflection credible?",
    ],
  };
  return common.concat(byArea[problem.area]);
}
