import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { test } from "node:test";
import { Client } from "@modelcontextprotocol/client";
import { InMemoryTransport } from "@modelcontextprotocol/server";

import { createServer } from "../src/server.js";
import { createFixtureRepository } from "./fixture.js";

test("MCP v2 client can list and call tutor tools", async () => {
  const root = createFixtureRepository();
  const server = createServer({ repositoryRoot: root });
  const client = new Client({ name: "aimlinterviews-mcp-test", version: "0.1.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    const listed = await client.listTools();
    assert.deepEqual(
      listed.tools.map((tool) => tool.name).sort(),
      [
        "get_company_prep",
        "get_hint",
        "get_learning_path",
        "get_problem",
        "get_server_status",
        "list_problems",
        "review_answer",
        "search_curriculum",
      ],
    );
    const called = await client.callTool({
      name: "list_problems",
      arguments: { area: "pytorch" },
    });
    assert.equal(called.isError, undefined);
    assert.match(textOf(called), /pytorch/i);
  } finally {
    await client.close();
    await server.close();
    rmSync(root, { recursive: true, force: true });
  }
});

function textOf(result: { content: Array<Record<string, unknown>> }): string {
  const block = result.content.find((item) => item.type === "text");
  return typeof block?.text === "string" ? block.text : "";
}
