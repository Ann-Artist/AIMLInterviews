import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { test } from "node:test";

import { buildCatalog } from "../src/catalog.js";
import { readRepositoryFile, resolveRepositoryRoot } from "../src/repository.js";
import { createFixtureRepository } from "./fixture.js";

test("catalog extracts prompts with stable IDs and omits reference answers", () => {
  const root = createFixtureRepository();
  try {
    const first = buildCatalog(root);
    const second = buildCatalog(root);
    assert.ok(first.problems.length >= 6);
    assert.deepEqual(first.problems.map((problem) => problem.id), second.problems.map((problem) => problem.id));
    assert.ok(first.problems.every((problem) => problem.id.startsWith("aiml:problem:")));
    assert.ok(first.problems.every((problem) => problem.sourceCommit === "unknown"));
    assert.equal(JSON.stringify(first).includes("SECRET_REFERENCE_IMPLEMENTATION"), false);
    assert.equal(JSON.stringify(first).includes("SECRET_INLINE_ANSWER"), false);
    assert.equal(first.problems.some((problem) => problem.area === "pytorch"), true);
    assert.equal(resolveRepositoryRoot(root), root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("repository reads cannot escape the configured root", () => {
  const root = createFixtureRepository();
  try {
    assert.throws(() => readRepositoryFile(root, "../outside.md"), /outside repository root/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
