export interface SemanticCategory {
  id: string;
  label: string;
  description: string;
  centroid: number[];
  emailCount: number;
  representativeEmails: string[];
  source: 'auto' | 'user';
  createdAt: number;
  updatedAt: number;
}

export interface CategorizationResult {
  messageId: string;
  matches: CategoryMatch[];
}

export interface CategoryMatch {
  categoryId: string;
  label: string;
  confidence: number;
  explanation?: string;
}

export interface ClusteringParams {
  minK: number;
  maxK: number;
  maxIterations: number;
  folders: string[];
}

export interface ClusteringResult {
  k: number;
  silhouetteScore: number;
  categories: SemanticCategory[];
  totalEmails: number;
  timestamp: number;
}
