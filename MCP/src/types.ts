import type { CallToolResult } from "@modelcontextprotocol/server";

export const AREAS = [
  "general-coding",
  "ml-coding",
  "pytorch",
  "ml-fundamentals",
  "genai",
  "system-design",
  "behavioral",
] as const;

export type Area = (typeof AREAS)[number];
export type Difficulty = "easy" | "medium" | "hard";

export interface Problem {
  id: string;
  title: string;
  prompt: string;
  area: Area;
  difficulty: Difficulty;
  tags: string[];
  companies: string[];
  sourcePath: string;
  sourceLine: number;
  sourceHeading: string;
  sourceCommit: string;
}

export interface CurriculumSection {
  id: string;
  title: string;
  area: Area;
  text: string;
  sourcePath: string;
  sourceLine: number;
  sourceCommit: string;
}

export interface CurriculumCatalog {
  repositoryRoot: string;
  sourceCommit: string;
  generatedAt: string;
  problems: Problem[];
  sections: CurriculumSection[];
}

export type ToolTextResult = CallToolResult;
