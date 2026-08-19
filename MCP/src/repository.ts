import { execFileSync } from "node:child_process";
import {
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

const REQUIRED_PATHS = [
  "README.md",
  "src/lc-coding.md",
  "src/MLC/ml-coding.md",
  "src/ml-fundamental.md",
  "src/MLSD/ml-system-design.md",
];

const BEHAVIOR_PATHS = ["src/behavior.md", "src/behavioral/behavior.md"];

const MAX_SOURCE_BYTES = 2 * 1024 * 1024;

export class RepositoryConfigurationError extends Error {}

export function resolveRepositoryRoot(explicitRoot?: string): string {
  const candidates = [explicitRoot, process.env.AIMLINTERVIEWS_ROOT, process.cwd()]
    .filter((candidate): candidate is string => Boolean(candidate))
    .map((candidate) => resolve(candidate));

  for (const candidate of candidates) {
    const discovered = findRepositoryRoot(candidate);
    if (discovered) return discovered;
  }

  throw new RepositoryConfigurationError(
    "AIMLInterviews repository not found. Clone https://github.com/alirezadir/AIMLInterviews and set AIMLINTERVIEWS_ROOT to the clone path.",
  );
}

export function findRepositoryRoot(start: string): string | null {
  let current = start;
  if (existsSync(current) && !statSync(current).isDirectory()) {
    current = dirname(current);
  }

  for (let depth = 0; depth < 10; depth += 1) {
    if (looksLikeRepository(current)) return current;
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

export function getSourceCommit(root: string): string {
  try {
    return execFileSync("git", ["-C", root, "rev-parse", "HEAD"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 2_000,
    }).trim();
  } catch {
    return "unknown";
  }
}

export function discoverMarkdownFiles(root: string): string[] {
  const files: string[] = [];
  walk(join(root, "src"), root, files);
  return files.sort();
}

export function readRepositoryFile(root: string, relativePath: string): string {
  const normalizedRoot = resolve(root);
  const fullPath = resolve(normalizedRoot, relativePath);
  const pathFromRoot = relative(normalizedRoot, fullPath);
  if (
    pathFromRoot.startsWith(`..${sep}`) ||
    pathFromRoot === ".." ||
    isAbsolute(pathFromRoot)
  ) {
    throw new Error(`Refusing to read outside repository root: ${relativePath}`);
  }
  if (!existsSync(fullPath) || !statSync(fullPath).isFile()) {
    throw new Error(`Repository file not found: ${relativePath}`);
  }
  if (statSync(fullPath).size > MAX_SOURCE_BYTES) {
    throw new Error(`Repository file exceeds ${MAX_SOURCE_BYTES} bytes: ${relativePath}`);
  }
  return readFileSync(fullPath, "utf8");
}

function looksLikeRepository(directory: string): boolean {
  return (
    REQUIRED_PATHS.every((path) => existsSync(join(directory, path))) &&
    BEHAVIOR_PATHS.some((path) => existsSync(join(directory, path)))
  );
}

function walk(directory: string, root: string, output: string[]): void {
  if (!existsSync(directory)) return;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const fullPath = join(directory, entry.name);
    if (entry.isSymbolicLink() || lstatSync(fullPath).isSymbolicLink()) continue;
    if (entry.isDirectory()) {
      walk(fullPath, root, output);
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      output.push(relative(root, fullPath).split(sep).join("/"));
    }
  }
}
