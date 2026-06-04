/**
 * Resolves which model and provider to use for user-facing chat calls, applying
 * cloud and local fallback rules so an unavailable or mismatched model is never used.
 */
import type { Logger } from 'pino';
import type { LlmConfig } from '../config/schema.js';
import type { LlmClient } from '../llm/client.js';
import { canonicalizeModelId } from './model-id.js';

/** Resolved chat model identity and the provider it should run against. */
export interface ActiveChatModel {
  modelId: string;
  provider: 'local' | 'cloud';
}

/** Compare two model ids for equality after canonicalizing their formats. */
function sameModel(a: string, b: string): boolean {
  return canonicalizeModelId(a) === canonicalizeModelId(b);
}

/**
 * Pick the model for user-facing assistant/chat calls.
 *
 * Cloud mode requires an explicit cloud chat model and never falls back to the local generation
 * model, which a cloud provider would reject. Local mode uses the configured local chat model only
 * when the local provider actually exposes it, so a stale cloud model name is not sent to Ollama.
 */
export async function resolveActiveChatModel(
  llm: LlmConfig,
  client: Pick<LlmClient, 'health'>,
  logger?: Pick<Logger, 'warn'>,
): Promise<ActiveChatModel> {
  if (llm.chatBaseUrl) {
    const cloudModel = llm.chatModel?.trim();
    if (!cloudModel) {
      throw new Error(
        'No cloud chat model is set. Open Settings and enter a Model (for example gpt-4o-mini) for the cloud provider.',
      );
    }
    return { modelId: canonicalizeModelId(cloudModel), provider: 'cloud' };
  }

  const requested = llm.chatModel?.trim() || llm.generationModel;
  const generationModel = canonicalizeModelId(llm.generationModel);
  if (!llm.chatModel?.trim() || sameModel(requested, generationModel)) {
    return { modelId: generationModel, provider: 'local' };
  }

  const health = await client.health();
  const available = health.models.some((model) => sameModel(model, requested));
  if (health.ok && available) {
    return { modelId: canonicalizeModelId(requested), provider: 'local' };
  }

  logger?.warn(
    { requested: canonicalizeModelId(requested), fallback: generationModel, healthOk: health.ok },
    'configured local chat model is unavailable; falling back to generation model',
  );
  return { modelId: generationModel, provider: 'local' };
}
