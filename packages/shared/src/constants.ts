/**
 * Shared application constants, including server endpoints, model presets,
 * embedding requirements, and clustering and retrieval defaults.
 */
import type { ModelPreset } from './types/config.js';

export const CORE_SERVER_PORT = 3420;
export const CORE_SERVER_HOST = 'localhost';
export const CORE_SERVER_URL = `http://${CORE_SERVER_HOST}:${CORE_SERVER_PORT}`;
export const OLLAMA_DEFAULT_URL = 'http://localhost:11434';
export const API_PREFIX = '/api/v1';

/**
 * Required dimension for embedding vectors stored in sqlite-vec.
 * All presets must use a 1024-dim model or insertion fails.
 */
export const REQUIRED_EMBEDDING_DIMENSIONS = 1024;

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'lightweight',
    label: 'Lightweight (smaller LLM)',
    description:
      'Smaller generation model for machines with limited RAM. Embedding still uses bge-m3 (required for 1024-dim storage).',
    embeddingModel: 'bge-m3',
    generationModel: 'qwen3:4b',
    minRamGb: 6,
  },
  {
    id: 'recommended',
    label: 'Recommended',
    description:
      'Best balance of quality and performance. Strong multilingual and structured output.',
    embeddingModel: 'bge-m3',
    generationModel: 'qwen3:8b',
    minRamGb: 8,
  },
  {
    id: 'institutional',
    label: 'Institutional (Mistral)',
    description:
      'Aligned with Mistral AI institutional partnership. Strong French language quality.',
    embeddingModel: 'bge-m3',
    generationModel: 'mistral:7b',
    minRamGb: 8,
  },
  {
    id: 'maximum',
    label: 'Maximum Quality',
    description: 'Highest accuracy. Requires 16 GB+ RAM.',
    embeddingModel: 'bge-m3',
    generationModel: 'qwen3:14b',
    minRamGb: 16,
  },
  {
    id: 'custom',
    label: 'Custom',
    description: 'Choose your own models from Ollama. Embedding model MUST output 1024 dimensions.',
    embeddingModel: '',
    generationModel: '',
    minRamGb: 0,
  },
];

export const CLUSTERING_DEFAULTS = {
  minK: 2,
  maxK: 20,
  maxIterations: 100,
  convergenceThreshold: 1e-6,
} as const;

export const EMBEDDING_BATCH_SIZE = 10;
export const MAX_TEXT_LENGTH = 8192;
export const REPRESENTATIVE_EMAIL_COUNT = 5;
export const TOP_K_CATEGORIES = 5;
export const RAG_TOP_K = 10;
