import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { captureEnv } from "../test-utils/env.js";
import { resolveApiKeyForProvider } from "./model-auth.js";
import {
  buildAzureOpenAIProvider,
  resolveAzureOpenAIEnv,
  resolveImplicitProviders,
} from "./models-config.providers.js";

const AZURE_ENV_KEYS = [
  "AZURE_OPENAI_API_KEY",
  "AZURE_OPENAI_ENDPOINT",
  "AZURE_OPENAI_DEPLOYMENT",
  "AZURE_OPENAI_API_VERSION",
];

describe("Azure OpenAI provider", () => {
  describe("resolveAzureOpenAIEnv", () => {
    it("returns null when no env vars are set", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("returns null when only endpoint is set (no key)", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com";

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("returns null when only key is set (no endpoint)", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("returns config when key + endpoint are set", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com";

      try {
        const result = resolveAzureOpenAIEnv();
        expect(result).toEqual({
          endpoint: "https://myresource.openai.azure.com",
          deployment: undefined,
          apiVersion: undefined,
        });
      } finally {
        envSnapshot.restore();
      }
    });

    it("includes deployment and api version when set", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com/";
      process.env.AZURE_OPENAI_DEPLOYMENT = "gpt-4o";
      process.env.AZURE_OPENAI_API_VERSION = "2024-10-21";

      try {
        const result = resolveAzureOpenAIEnv();
        expect(result).toEqual({
          endpoint: "https://myresource.openai.azure.com",
          deployment: "gpt-4o",
          apiVersion: "2024-10-21",
        });
      } finally {
        envSnapshot.restore();
      }
    });

    it("rejects non-HTTPS endpoints", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "http://myresource.openai.azure.com";

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("rejects SSRF targets (metadata endpoint)", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://169.254.169.254";

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("rejects localhost endpoints", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://localhost";

      try {
        expect(resolveAzureOpenAIEnv()).toBeNull();
      } finally {
        envSnapshot.restore();
      }
    });

    it("sanitizes deployment names with path traversal", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com";
      process.env.AZURE_OPENAI_DEPLOYMENT = "../../../etc/passwd";

      try {
        const result = resolveAzureOpenAIEnv();
        // Deployment with path traversal chars is rejected (undefined)
        expect(result?.deployment).toBeUndefined();
      } finally {
        envSnapshot.restore();
      }
    });

    it("allows valid deployment names with dots and hyphens", () => {
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com";
      process.env.AZURE_OPENAI_DEPLOYMENT = "gpt-4o.2024-11";

      try {
        const result = resolveAzureOpenAIEnv();
        expect(result?.deployment).toBe("gpt-4o.2024-11");
      } finally {
        envSnapshot.restore();
      }
    });
  });

  describe("buildAzureOpenAIProvider", () => {
    it("uses v1 base URL when no api-version is set", () => {
      const provider = buildAzureOpenAIProvider({
        endpoint: "https://myresource.openai.azure.com",
        deployment: "gpt-4o",
      });

      expect(provider.baseUrl).toBe(
        "https://myresource.openai.azure.com/openai/v1",
      );
      expect(provider.api).toBe("openai-completions");
      expect(provider.models).toHaveLength(1);
      expect(provider.models[0].id).toBe("gpt-4o");
    });

    it("does not include secrets in headers or authHeader flag", () => {
      const provider = buildAzureOpenAIProvider({
        endpoint: "https://myresource.openai.azure.com",
        deployment: "gpt-4o",
      });

      // No headers with secrets; no authHeader override
      expect(provider.headers).toBeUndefined();
      expect(provider.authHeader).toBeUndefined();
    });

    it("uses legacy deployment URL when api-version is set", () => {
      const provider = buildAzureOpenAIProvider({
        endpoint: "https://myresource.openai.azure.com",
        deployment: "gpt-4o",
        apiVersion: "2024-10-21",
      });

      expect(provider.baseUrl).toBe(
        "https://myresource.openai.azure.com/openai/deployments/gpt-4o",
      );
    });

    it("falls back to v1 URL when api-version set but no deployment", () => {
      const provider = buildAzureOpenAIProvider({
        endpoint: "https://myresource.openai.azure.com",
        apiVersion: "2024-10-21",
      });

      expect(provider.baseUrl).toBe(
        "https://myresource.openai.azure.com/openai/v1",
      );
    });

    it("uses gpt-4o as default model when no deployment specified", () => {
      const provider = buildAzureOpenAIProvider({
        endpoint: "https://myresource.openai.azure.com",
      });

      expect(provider.models[0].id).toBe("gpt-4o");
      expect(provider.models[0].name).toBe("Azure OpenAI gpt-4o");
    });
  });

  describe("implicit provider registration", () => {
    it("includes azure-openai when env vars are set", async () => {
      const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "https://myresource.openai.azure.com";
      process.env.AZURE_OPENAI_DEPLOYMENT = "gpt-4o";

      try {
        const providers = await resolveImplicitProviders({ agentDir });
        expect(providers?.["azure-openai"]).toBeDefined();
        expect(providers?.["azure-openai"]?.api).toBe("openai-completions");
        // apiKey should be the env var name, NOT the actual secret
        expect(providers?.["azure-openai"]?.apiKey).toBe("AZURE_OPENAI_API_KEY");
        // No secrets in headers
        expect(providers?.["azure-openai"]?.headers).toBeUndefined();
        expect(providers?.["azure-openai"]?.models?.[0]?.id).toBe("gpt-4o");
      } finally {
        envSnapshot.restore();
      }
    });

    it("does not include azure-openai when env vars are missing", async () => {
      const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }

      try {
        const providers = await resolveImplicitProviders({ agentDir });
        expect(providers?.["azure-openai"]).toBeUndefined();
      } finally {
        envSnapshot.restore();
      }
    });

    it("does not include azure-openai when endpoint is invalid", async () => {
      const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
      const envSnapshot = captureEnv(AZURE_ENV_KEYS);
      for (const key of AZURE_ENV_KEYS) {
        delete process.env[key];
      }
      process.env.AZURE_OPENAI_API_KEY = "test-key";
      process.env.AZURE_OPENAI_ENDPOINT = "http://insecure.example.com";

      try {
        const providers = await resolveImplicitProviders({ agentDir });
        expect(providers?.["azure-openai"]).toBeUndefined();
      } finally {
        envSnapshot.restore();
      }
    });
  });

  describe("API key resolution", () => {
    it("resolves the azure-openai api key from env", async () => {
      const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
      const envSnapshot = captureEnv(["AZURE_OPENAI_API_KEY"]);
      process.env.AZURE_OPENAI_API_KEY = "azure-test-key";

      try {
        const auth = await resolveApiKeyForProvider({
          provider: "azure-openai",
          agentDir,
        });

        expect(auth.apiKey).toBe("azure-test-key");
        expect(auth.mode).toBe("api-key");
        expect(auth.source).toContain("AZURE_OPENAI_API_KEY");
      } finally {
        envSnapshot.restore();
      }
    });
  });
});
