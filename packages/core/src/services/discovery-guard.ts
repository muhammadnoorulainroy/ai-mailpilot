/**
 * Local-only guard for the category discovery and improvement flows. Discovery reads real
 * email content, so by default it must run against the local model and never the cloud. One
 * shared switch (allowCloudDiscovery) governs both topic discovery and Improve Categories.
 */
import type { LlmConfig } from '../config/schema.js';

/**
 * The provider discovery may use. Local unless the user has explicitly opted in to cloud
 * discovery and a cloud chat endpoint is configured.
 */
export function discoveryProvider(cfg: LlmConfig): 'main' | 'chat' {
  return cfg.allowCloudDiscovery && !!cfg.chatBaseUrl ? 'chat' : 'main';
}

/**
 * Throw if a discovery LLM call would leave the machine without an explicit opt-in. Call this
 * immediately before any topic discovery or category improvement chat request.
 */
export function assertDiscoveryLocal(cfg: LlmConfig, provider: 'main' | 'chat'): void {
  if (provider === 'chat' && !cfg.allowCloudDiscovery) {
    throw new Error(
      'Category discovery is local-only by default. Enable allowCloudDiscovery to send discovery to the cloud.',
    );
  }
}
