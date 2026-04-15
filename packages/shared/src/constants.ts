import type { ModelPreset } from './types/config.js';

export const CORE_SERVER_PORT = 3420;
export const CORE_SERVER_HOST = 'localhost';
export const CORE_SERVER_URL = `http://${CORE_SERVER_HOST}:${CORE_SERVER_PORT}`;
export const OLLAMA_DEFAULT_URL = 'http://localhost:11434';
export const API_PREFIX = '/api/v1';

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'lightweight',
    label: 'Lightweight',
    description: 'For machines with limited resources (4 GB RAM). Faster but less accurate.',
    embeddingModel: 'nomic-embed-text',
    generationModel: 'phi3.5:3.8b',
    minRamGb: 4,
  },
  {
    id: 'recommended',
    label: 'Recommended',
    description: 'Best balance of quality and performance.',
    embeddingModel: 'bge-m3',
    generationModel: 'mistral:7b',
    minRamGb: 8,
  },
  {
    id: 'maximum',
    label: 'Maximum Quality',
    description: 'Highest accuracy. Requires 16 GB+ RAM.',
    embeddingModel: 'bge-m3',
    generationModel: 'mistral-nemo:12b',
    minRamGb: 16,
  },
  {
    id: 'custom',
    label: 'Custom',
    description: 'Choose your own models from Ollama.',
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
