export interface EmailEmbedding {
  messageId: string;
  modelId: string;
  vector: number[];
  dimensions: number;
  createdAt: number;
}

export interface SimilarityResult {
  messageId: string;
  score: number;
}

export interface EmbeddingProgress {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  folder: string;
  total: number;
  processed: number;
  skipped: number;
  error?: string;
}
